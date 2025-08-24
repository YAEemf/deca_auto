"""
SPICEモデル -> PySpice で AC サンプリング -> scikit-rf VectorFitting (Z型)
フォールバックは rlc_model に委譲（build_capacitor_model 内で捕捉）

- parse_first_subckt(model_path) -> (subckt_name, [pin1, pin2])
- sample_impedance_with_pyspice(model_path, f_cpu, dc_bias) -> (f_s_cpu, Zc_s_cpu)
- fit_with_vector_fitting(f_cpu, Zc_cpu, n_poles_real=4, n_poles_cmplx=4) -> Callable[[xp.ndarray], xp.ndarray]
- build_capacitor_model(cap: CapSpec, f_dev: xp.ndarray, xp) -> xp.ndarray[complex]
  * 成功時: GPU 常駐 Zc(f_dev) を返す
  * 失敗時: rlc_model.analytic_Zc() を呼び出してフォールバック
例外は上位で traceback.print_exc() に委ねる前提
"""
from __future__ import annotations

import re
import pathlib
import numpy as np
import traceback
from typing import Callable, Tuple, List, Optional

from .config import CapSpec
from . import rlc_model

# PySpice
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_Hz, u_V
from PySpice.Spice.HighLevelElement import SinusoidalCurrentSource

_SUBCKT_RE = re.compile(
    r"(?i)^\s*\.SUBCKT\s+(\S+)\s+([^\n\r]+)\s*$", re.MULTILINE
)

def parse_first_subckt(model_path: str) -> Tuple[str, List[str]]:
    """
    .SUBCKT を正規表現でパースし、最初に現れる 2 端子サブ回路の (name, pins) を返す
    ピン数が 2 でない場合は ValueError
    """
    p = pathlib.Path(model_path).expanduser().resolve()
    text = p.read_text(encoding="utf-8", errors="ignore")
    for m in _SUBCKT_RE.finditer(text):
        name = m.group(1)
        pins = re.split(r"\s+", m.group(2).strip())
        if len(pins) == 2:
            return name, pins
    raise ValueError("2端子の .SUBCKT 定義が見つかりません。")


def _log_interp1d(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """log-log 線形補間（端は外挿せず端値保持）"""
    lx, lxp, lyp = np.log10(x_new), np.log10(x), np.log10(np.abs(y))
    mag = np.power(10.0, np.interp(lx, lxp, lyp))
    # 位相は線形補間（unwrap）
    ph = np.unwrap(np.angle(y))
    ph_i = np.interp(x_new, x, ph)
    return mag * np.exp(1j * ph_i)


def sample_impedance_with_pyspice(
    model_path: str, f_cpu: np.ndarray, dc_bias: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PySpice で AC サンプリング
    回路: GND -- Iac(1A) -- (n1) -- X(dut) -- GND
    - .include に model_path を渡し、最初の 2端子 .SUBCKT を DUT とする
    - AC 解析は probes=['n1'] 指定が必須（PySpice 仕様）
    - Iac は HighLevelElement.SinusoidalCurrentSource(ac_magnitude=1) を使用（ac 値必須）
    出力:
      f_s_cpu: サンプル周波数（log sweep）
      Zc_s_cpu: サンプル Zc（V/I、I=1Aなので V=Z）
    """
    # 2端子 SUBCKT 名とピン名を取得
    subckt_name, pins = parse_first_subckt(model_path)

    # 回路生成
    circuit = Circuit("Zc_measure")
    # モデル読み込み（相対パスにも対応）
    p = pathlib.Path(model_path).expanduser().resolve()
    circuit.include(str(p))

    # ノード名
    n1 = "n1"
    # 独立電流源（AC=1A）
    # ac_magnitude を指定して AC 解析で 1A とする
    SinusoidalCurrentSource(circuit, "IAC", n1, circuit.gnd, ac_magnitude=1)

    # DUT インスタンス（Xname）
    # pin順は [node_plus, node_minus] を想定
    circuit.X("DUT", subckt_name, n1, circuit.gnd)

    # シミュレータ生成
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    try:
        # 補助: 生 netlist を保存（デバッグ用）
        netlist_path = p.with_name(f"last_fail_{subckt_name}.cir")
        with netlist_path.open("w", encoding="utf-8") as f:
            f.write(str(circuit))

        # probes 保存（v(n1) を含める）
        simulator.save(["all", f"v({n1})"]) 

        # PySpice の ac は対数掃引をサポート（variation='dec'）
        f_start = float(np.min(f_cpu))
        f_stop = float(np.max(f_cpu))
        npt = max(32, len(f_cpu))  # 解析点が少なすぎると不安定
        analysis = simulator.ac(
            start_frequency=f_start @ u_Hz,
            stop_frequency=f_stop @ u_Hz,
            number_of_points=npt,
            variation="dec",
            probes=[n1],  # AC/Transient は probes 必須
        )

        # 取り出し（Iac=1A なので Z=V）
        v_n1 = np.array(analysis[n1])  # complex np.ndarray
        f_s = np.array(analysis.frequency)  # Hz

        # 必要なら log 補間で f_cpu に合わせる
        if len(f_s) != len(f_cpu) or np.any(f_s != f_cpu):
            Zc = _log_interp1d(f_cpu, f_s, v_n1)
            return f_cpu, Zc
        else:
            return f_s, v_n1
    except Exception:
        # 例外は上位で出すが、ここでもネットリストが出力済み
        raise


def fit_with_vector_fitting(f_s: np.ndarray, Zc_s: np.ndarray):
    """
    scikit-rf VectorFitting による z-パラメータ近似
    - f_s: 1D 周波数 [Hz]
    - Zc_s: 1D 複素インピーダンス（1ポート）
    戻り値: eval_fn(f_new[Hz]) -> Z_fit(np.ndarray)
    """
    import numpy as _np
    import skrf as rf
    from skrf.vectorFitting import VectorFitting

    # 1ポート Z → (F,1,1) に整形
    z = _np.asarray(Zc_s, dtype=_np.complex128).reshape(-1, 1, 1)
    f_arr = _np.asarray(f_s, dtype=float)

    # 互換性重視：まずは Network(z=z, frequency=Frequency) を試す
    try:
        fobj = rf.Frequency.from_f(f_arr, unit="Hz")
        ntw = rf.Network(z=z, frequency=fobj)
    except Exception:
        # 古い/環境差のある場合は from_z に数値配列 f を渡す
        ntw = rf.Network.from_z(z, f=f_arr)

    # VF 本体（表現タイプは vector_fit 側で指定）
    vf = VectorFitting(ntw)
    vf.vector_fit(
        n_poles_real=2,
        n_poles_cmplx=4,
        parameter_type="z",   # z-パラメータを近似する指定（コンストラクタではない）
        fit_constant=True,
        fit_proportional=False,
    )

    def eval_fn(f_new_hz: _np.ndarray) -> _np.ndarray:
        freqs = _np.asarray(f_new_hz, dtype=float)
        # VF モデルから z11 を取得
        return _np.asarray(vf.get_model_response(i=0, j=0, freqs=freqs))

    return eval_fn


def build_capacitor_model(cap: CapSpec, f_dev, xp) -> "xp.ndarray":
    """
    コンデンサ1種の Zc(f) を生成し GPU 常駐で返す
    優先: PySpice -> VectorFitting（Z型）
    フォールバック: 解析式（R+L+理想C+L_mnt）を rlc_model.analytic_Zc で計算
    """
    try:
        # PySpice は CPU 上で動作するため f を CPU に取り出す
        f_cpu = np.asarray(f_dev.get() if hasattr(f_dev, "get") else f_dev)
        if cap.model_path:
            f_s, Zc_s = sample_impedance_with_pyspice(cap.model_path, f_cpu, dc_bias=0.0)  # DC バイアスは AC 動作点のみ
            # VF で連続モデル化して f_cpu で再評価
            eval_fn = fit_with_vector_fitting(f_s, Zc_s)
            Z_fit_cpu = eval_fn(f_cpu)
            return xp.asarray(Z_fit_cpu, dtype=getattr(xp, "complex64", None))
    except Exception:
        # 失敗時は上位で traceback.print_exc()、ここではそのままフォールバック
        traceback.print_exc()

    # --- フォールバック（RLC 解析式） ---
    C = cap.C if cap.C is not None else 0.0
    return rlc_model.analytic_Zc(
        f_dev=f_dev, C=C, ESR=cap.ESR, ESL=cap.ESL, Lmnt=(cap.L_mnt or 0.0),
        xp=xp, dtype_c=getattr(xp, "complex64", None),
    )

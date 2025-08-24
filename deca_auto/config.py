"""
ユーザー設定の既定値定義・TOML読み込み・検証
- load_user_config(): TOML が指すキーのみ上書きして dict を返す
- validate_config(): 値検証と正規化を行い、Config(dataclass) を返す
他モジュール互換:
  backend.select_xp(), freqgrid.make_freq_grid(), spice_model.build_capacitor_model(),
  rlc_model.analytic_Zc(), pdn.*, score.*, plotter.*, excel_out.* が参照するキーを網羅
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import math
import pathlib
import sys
import traceback

import tomllib

@dataclass(frozen=True)
class CapSpec:
    """コンデンサ種別の定義（SPICE / 解析式フォールバック）"""
    name: str
    model_path: Optional[str] = None  # None のとき解析式
    C: Optional[float] = None         # model_path があれば省略可
    ESR: float = 20e-3
    ESL: float = 1e-9
    L_mnt: Optional[float] = None     # None の場合は cfg.L_mntN を使用


@dataclass(frozen=True)
class Config:
    toml_path: Optional[str]

    f_start: float
    f_stop: float
    num_points: int

    f_L: float
    f_H: float

    z_target: float
    z_custom_mask: Optional[List[Tuple[float, float]]]

    R_vrm: float
    L_vrm: float
    R_sN: float
    L_sN: float
    L_mntN: float
    R_s: float
    L_s: float
    R_v: float
    L_v: float
    R_p: float
    C_p: float
    tan_delta_p: float

    dc_bias: float

    capacitors: List[CapSpec]

    max_total_parts: int
    min_total_parts_ratio: float
    top_k: int
    shuffle_evaluation: bool

    weights: Dict[str, float]

    mc_enable: bool
    mc_samples: int
    tol_C: float
    tol_ESR: float
    tol_ESL: float
    mlcc_derating: float

    seed: int

    max_vram_ratio: float
    dtype_c: str
    dtype_r: str
    force_numpy: bool

    plot_view: bool
    plot_min_interval_s: float
    plot_force_update_interval_s: float

    excel_path: Optional[str]
    excel_name: str


DEFAULT_USER_CONFIG: Dict[str, Any] = {
    # TOML のパス（情報）
    "toml_path": "user_config.toml",

    # 周波数グリッド
    "f_start": 1e2, "f_stop": 1e9, "num_points": 1024,

    # 評価帯域
    "f_L": 1e3, "f_H": 1e8, "z_target": 10e-3,
    "z_custom_mask": [
        [1e3, 10e-3], [1e5, 10e-3], [2e6, 30e-3], [1e8, 1.5]
    ],

    # PDN 寄生成分
    "L_mntN": 1e-9,
    "tan_delta_p": 0.02,
    "R_vrm": 10e-3, "L_vrm": 10e-9,
    "R_sN": 0.5e-3, "L_sN": 1e-9,
    "R_s": 0.5e-3, "L_s": 1e-9,
    "R_v": 0.5e-3, "L_v": 1e-9,
    "R_p": 0.5e-3, "C_p": 5e-12,

    # PySpice 用 DC バイアス
    "dc_bias": 3.3,

    # コンデンサリスト
    "capacitors": [
        {"name": "C_0603_0u1_default", "C": 0.1e-6, "ESR": 20e-3, "ESL": 0.8e-9},
    ],

    # 組合せ探索
    "max_total_parts": 16,
    "min_total_parts_ratio": 0.3,
    "top_k": 10,
    "shuffle_evaluation": True,

    # スコア重み
    "weights": {
        "max": 0.9, "area": 0.75, "anti": 0.35, "flat": 0.15,
        "under": -0.1, "parts": 0.0, "mc_worst": 1.0
    },

    # Monte Carlo
    "mc_enable": True, "mc_samples": 64,
    "tol_C": 0.2, "tol_ESR": 0.2, "tol_ESL": 0.2,
    "mlcc_derating": 0.15,

    # 乱数シード
    "seed": 1234,

    # GPU関連
    "max_vram_ratio": 0.1,
    "dtype_c": "complex64",
    "dtype_r": "float32",
    "force_numpy": False,

    # グラフ関連
    "plot_view": True,
    "plot_min_interval_s": 0.5,
    "plot_force_update_interval_s": 5.0,

    # Excel
    "excel_path": None,
    "excel_name": "dcap_result",
}

def load_user_config(toml_path: Optional[str]) -> Dict[str, Any]:
    """
    TOML のキーのみ DEFAULT_USER_CONFIG に上書きして dict を返す
    - toml_path が None またはファイル無しなら DEFAULT をそのまま返す
    例外は上位で traceback.print_exc() される前提
    """
    cfg = DEFAULT_USER_CONFIG.copy()
    if not toml_path:
        return cfg

    try:
        p = pathlib.Path(toml_path).expanduser()
        if p.is_file():
            with p.open("rb") as f:
                toml_data = tomllib.load(f)
            # TOML に存在するキーのみ上書き
            for k, v in toml_data.items():
                if k in cfg:
                    cfg[k] = v
        # ファイルが無い場合は既定を使用（通知は main 側でログ）
    except Exception:
        traceback.print_exc()
    return cfg


def validate_config(cfg: Dict[str, Any]) -> Config:
    """
    値検証・正規化・型付け（Config dataclass を返す）
    - カスタム目標マスク指定時：f_L/f_H はマスクの min/max で上書き
    - 各コンデンサの省略値（ESR/ESL/L_mnt）を規約通り適用
    """
    # 周波数グリッド
    f_start = float(cfg["f_start"])
    f_stop = float(cfg["f_stop"])
    num_points = int(cfg["num_points"])
    if not (f_start > 0 and f_stop > f_start and num_points >= 16):
        raise ValueError("周波数グリッド設定が不正です。")

    # 評価帯域とマスク
    z_custom_mask = cfg.get("z_custom_mask")
    f_L = float(cfg["f_L"])
    f_H = float(cfg["f_H"])
    z_target = float(cfg["z_target"])
    if z_custom_mask:
        # 折れ線点の検証とソート
        pts = []
        for it in z_custom_mask:
            f, z = float(it[0]), float(it[1])
            if f <= 0 or z <= 0:
                raise ValueError("z_custom_mask の周波数/インピーダンスは正である必要があります。")
            pts.append((f, z))
        pts.sort(key=lambda x: x[0])
        z_custom_mask = pts
        f_L, f_H = pts[0][0], pts[-1][0]

    if not (f_start <= f_L < f_H <= f_stop):
        raise ValueError("評価帯域 f_L/f_H は f_start/f_stop の範囲に含まれる必要があります。")

    # PDN 寄生
    def _pos(x, name):
        x = float(x)
        if x < 0:
            raise ValueError(f"{name} は負であってはなりません。")
        return x

    R_vrm = _pos(cfg["R_vrm"], "R_vrm")
    L_vrm = _pos(cfg["L_vrm"], "L_vrm")
    R_sN = _pos(cfg["R_sN"], "R_sN")
    L_sN = _pos(cfg["L_sN"], "L_sN")
    L_mntN = _pos(cfg["L_mntN"], "L_mntN")
    R_s = _pos(cfg["R_s"], "R_s")
    L_s = _pos(cfg["L_s"], "L_s")
    R_v = _pos(cfg["R_v"], "R_v")
    L_v = _pos(cfg["L_v"], "L_v")
    R_p = _pos(cfg["R_p"], "R_p")
    C_p = _pos(cfg["C_p"], "C_p")
    tan_delta_p = _pos(cfg["tan_delta_p"], "tan_delta_p")
    dc_bias = float(cfg["dc_bias"])

    # コンデンサ
    caps_raw = cfg.get("capacitors", [])
    capacitors: List[CapSpec] = []
    for item in caps_raw:
        name = str(item["name"])
        model_path = item.get("model_path")
        C = item.get("C", None)
        ESR = float(item.get("ESR", 20e-3))
        ESL = float(item.get("ESL", 1e-9))
        L_mnt = item.get("L_mnt", None)
        if C is not None:
            C = float(C)
            if C <= 0:
                raise ValueError(f"{name}: C は正である必要があります。")
        if L_mnt is not None:
            L_mnt = float(L_mnt)

        capacitors.append(
            CapSpec(
                name=name, model_path=model_path, C=C,
                ESR=ESR, ESL=ESL, L_mnt=L_mnt
            )
        )

    # 探索境界
    max_total_parts = int(cfg["max_total_parts"])
    min_total_parts_ratio = float(cfg["min_total_parts_ratio"])
    if not (max_total_parts >= 1 and 0.0 <= min_total_parts_ratio <= 1.0):
        raise ValueError("max_total_parts / min_total_parts_ratio が不正です。")
    if int(cfg["top_k"]) < 1:
        raise ValueError("top_k は 1 以上。")
    top_k = int(cfg["top_k"])
    shuffle_evaluation = bool(cfg["shuffle_evaluation"])

    # スコア重み（必要キー存在チェック）
    weights = dict(cfg["weights"])
    for k in ("max", "area", "anti", "flat", "under", "parts", "mc_worst"):
        if k not in weights:
            raise ValueError(f"weights に {k} が不足しています。")

    # Monte Carlo
    mc_enable = bool(cfg["mc_enable"])
    mc_samples = int(cfg["mc_samples"])
    tol_C = float(cfg["tol_C"])
    tol_ESR = float(cfg["tol_ESR"])
    tol_ESL = float(cfg["tol_ESL"])
    mlcc_derating = float(cfg["mlcc_derating"])
    seed = int(cfg["seed"])

    # backend・精度
    max_vram_ratio = float(cfg["max_vram_ratio"])
    if not (0.0 < max_vram_ratio <= 1.0):
        raise ValueError("max_vram_ratio は 0–1 の範囲。")
    dtype_c = str(cfg["dtype_c"])
    dtype_r = str(cfg["dtype_r"])
    force_numpy = bool(cfg["force_numpy"])

    # plot / excel
    plot_view = bool(cfg["plot_view"])
    plot_min_interval_s = float(cfg["plot_min_interval_s"])
    plot_force_update_interval_s = float(cfg["plot_force_update_interval_s"])
    excel_path = cfg["excel_path"]
    if excel_path is not None:
        excel_path = str(excel_path)
    excel_name = str(cfg["excel_name"])

    # L_mnt が未指定の CapSpec に対して既定 L_mntN を適用
    caps_resolved: List[CapSpec] = []
    for c in capacitors:
        caps_resolved.append(
            CapSpec(
                name=c.name, model_path=c.model_path, C=c.C,
                ESR=c.ESR, ESL=c.ESL, L_mnt=(c.L_mnt if c.L_mnt is not None else L_mntN)
            )
        )

    return Config(
        toml_path=cfg.get("toml_path"),

        f_start=f_start, f_stop=f_stop, num_points=num_points,
        f_L=f_L, f_H=f_H,
        z_target=z_target, z_custom_mask=z_custom_mask,

        R_vrm=R_vrm, L_vrm=L_vrm,
        R_sN=R_sN, L_sN=L_sN, L_mntN=L_mntN,
        R_s=R_s, L_s=L_s, R_v=R_v, L_v=L_v,
        R_p=R_p, C_p=C_p, tan_delta_p=tan_delta_p,

        dc_bias=dc_bias,
        capacitors=caps_resolved,

        max_total_parts=max_total_parts,
        min_total_parts_ratio=min_total_parts_ratio,
        top_k=top_k,
        shuffle_evaluation=shuffle_evaluation,

        weights=weights,

        mc_enable=mc_enable, mc_samples=mc_samples,
        tol_C=tol_C, tol_ESR=tol_ESR, tol_ESL=tol_ESL,
        mlcc_derating=mlcc_derating,

        seed=seed,

        max_vram_ratio=max_vram_ratio,
        dtype_c=dtype_c, dtype_r=dtype_r, force_numpy=force_numpy,

        plot_view=plot_view,
        plot_min_interval_s=plot_min_interval_s,
        plot_force_update_interval_s=plot_force_update_interval_s,

        excel_path=excel_path, excel_name=excel_name,
    )

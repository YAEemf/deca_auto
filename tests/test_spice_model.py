# -*- coding: utf-8 -*-
"""
SPICE モデル（任意）または RLC フォールバックで |Zc| を両対数描画・保存
- 既定では RLC フォールバック（PySpice 不要）
- --model PATH を渡すと、その SPICE .mod/.lib を PySpice+VF で評価（失敗時は自動でフォールバック）
- 保存先: tests/output/zc_models.png
実行:
  1) 直接実行:  python tests/test_spice_model.py --toml user_config.toml
  2) pytest:     pytest -q tests/test_spice_model.py
"""

from __future__ import annotations

import argparse
import pathlib
import traceback
import numpy as np
import matplotlib.pyplot as plt

from deca_auto import config as cfgmod
from deca_auto.backend import select_xp
from deca_auto.freqgrid import make_freq_grid
from deca_auto.spice_model import build_capacitor_model
from deca_auto.rlc_model import analytic_Zc


def _ensure_outdir() -> pathlib.Path:
    out_dir = pathlib.Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def test_plot_spice_model(toml: str | None = None, model_path: str | None = None):
    try:
        # 設定ロード・検証
        raw = cfgmod.load_user_config(toml)
        # テスト用に最低限のコンデンサを差し込む（toml の capacitors が空でも動くように）
        if not raw.get("capacitors"):
            raw["capacitors"] = [
                {"name": "C_0603_0u1", "C": 0.1e-6},
                {"name": "C_0603_1u0", "C": 1.0e-6},
            ]
        if model_path:
            # 追加で SPICE モデル 1 種を評価（pin は 2 端子の想定）
            raw["capacitors"].append({"name": "SPICE_CAP", "model_path": str(model_path)})

        cfg = cfgmod.validate_config(raw)

        # バックエンド
        xp, _ = select_xp(force_numpy=cfg.force_numpy)

        # 周波数グリッド
        f_dev = make_freq_grid(cfg, xp)
        f_cpu = np.asarray(f_dev.get() if hasattr(f_dev, "get") else f_dev)

        # 各コンデンサの Zc を算出
        Zc_cpu_dict = {}
        for cap in cfg.capacitors:
            try:
                if cap.model_path:
                    z_dev = build_capacitor_model(cap, f_dev, xp)  # PySpice+VF 失敗時は内部でフォールバック
                else:
                    z_dev = analytic_Zc(f_dev, C=cap.C, ESR=cap.ESR, ESL=cap.ESL, Lmnt=cap.L_mnt, xp=xp, dtype_c=getattr(xp, cfg.dtype_c))
            except Exception:
                traceback.print_exc()
                # どのみちフォールバック
                z_dev = analytic_Zc(f_dev, C=(cap.C or 0.0), ESR=cap.ESR, ESL=cap.ESL, Lmnt=cap.L_mnt, xp=xp, dtype_c=getattr(xp, cfg.dtype_c))
            Zc_cpu_dict[cap.name] = np.asarray(z_dev.get() if hasattr(z_dev, "get") else z_dev)

        # 描画（両対数）
        plt.figure(figsize=(7, 4))
        ax = plt.gca()
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.4)
        ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("|Zc| [Ohm]")
        for name, z in Zc_cpu_dict.items():
            ax.plot(f_cpu, np.abs(z), lw=1.2, label=name)
        if ax.has_data():
            ax.legend(loc="upper left", fontsize=8, framealpha=0.5)
            ax.set_xlim(float(f_cpu[0]), float(f_cpu[-1]))

        out = _ensure_outdir() / "zc_models.png"
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved: {out}")
    except Exception:
        traceback.print_exc()
        assert False, "test_plot_spice_model failed"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--toml", type=str, default=None)
    ap.add_argument("--model", type=str, default=None, help="Optional SPICE model path (.mod/.lib)")
    args = ap.parse_args()
    test_plot_spice_model(args.toml, args.model)

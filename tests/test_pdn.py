# -*- coding: utf-8 -*-
"""
3 種のコンデンサ [0.1uF, 1uF, 10uF] で PDN を組み、|Z_in| を両対数表示
  * pdn.zin_batch() で (B,F) の Z_in を一括計算
  * 「段順序は最終ノード（Load側）から見て小容量 → 大容量」：pdn.sort_indices_by_C_small_to_large()
  * 直列/並列合成は pdn.build_static_parasitics() / pdn.zin_batch() 内で実装済み
- 実行モードは test_spice_model.py と同様
"""

from __future__ import annotations

import os
import pathlib
import traceback
import numpy as np
import matplotlib
if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deca_auto import config as cfgmod
from deca_auto.backend import select_xp
from deca_auto.freqgrid import make_freq_grid, make_eval_band_and_mask
from deca_auto.spice_model import build_capacitor_model
from deca_auto.pdn import build_static_parasitics, sort_indices_by_C_small_to_large, zin_batch


def _ensure_outdir() -> pathlib.Path:
    outdir = pathlib.Path(__file__).resolve().parent / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def test_pdn_three_caps():
    """[0.1u, 1u, 10u] → |Z_in| を作図・保存（2通りのカウントベクトル）"""
    try:
        # 1) 設定
        user = cfgmod.DEFAULT_USER_CONFIG.copy()
        user["force_numpy"] = True
        user["plot_view"] = False
        user["capacitors"] = [
            {"name": "C_0603_0u1", "C": 0.1e-6, "ESR": 20e-3, "ESL": 1.0e-9},
            {"name": "C_0603_1u",  "C": 1.0e-6, "ESR": 20e-3, "ESL": 1.0e-9},
            {"name": "C_0805_10u", "C": 10e-6,  "ESR": 30e-3, "ESL": 1.2e-9},
        ]
        cfg = cfgmod.validate_config(user)

        xp, _ = select_xp(force_numpy=cfg.force_numpy)
        dtype_c = getattr(xp, cfg.dtype_c)

        # 2) 周波数・評価帯域
        f_dev = make_freq_grid(cfg, xp)
        f_cpu = np.asarray(f_dev.get() if hasattr(f_dev, "get") else f_dev)
        m_eval, z_target = make_eval_band_and_mask(cfg, f_dev, xp)

        # 3) 各 Z_c
        Zc_list = [build_capacitor_model(cap, f_dev, xp) for cap in cfg.capacitors]

        # 4) 段順序 & 寄生
        order_idx = sort_indices_by_C_small_to_large([c.C for c in cfg.capacitors])
        parasitics = build_static_parasitics(f_dev, cfg, xp, dtype_c)

        # 5) 2 つの組合せ（B=2）
        counts_cpu = np.asarray([
            [1, 1, 1],   # 3 個
            [2, 3, 1],   # 6 個
        ], dtype=np.int16)
        counts_dev = xp.asarray(counts_cpu)

        # 6) Z_in 計算（B,F）
        Zin = zin_batch(
            f_dev=f_dev,
            Zc_list=Zc_list,
            counts_dev=counts_dev,
            parasitics=parasitics,
            order_idx=order_idx,
            cap_specs=cfg.capacitors,
            xp=xp,
            dtype_c=dtype_c,
        )

        Zin_cpu = np.asarray(Zin.get() if hasattr(Zin, "get") else Zin)

        # 7) プロット
        outdir = _ensure_outdir()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.4)
        ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("|Zin| [Ohm]")
        labels = ["(0.1u)x1 + (1u)x1 + (10u)x1 (Total:3)", "(0.1u)x2 + (1u)x3 + (10u)x1 (Total:6)"]
        for i in range(Zin_cpu.shape[0]):
            ax.plot(f_cpu, np.abs(Zin_cpu[i]), lw=1.2, label=labels[i])
        ax.legend(loc="upper left", fontsize=8, framealpha=0.5)
        ax.set_xlim(float(f_cpu[0]), float(f_cpu[-1]))
        fig.tight_layout()
        png = outdir / "test_pdn_three_caps.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)

        # 8) 簡易な健全性チェック（評価帯域で有限値）
        assert np.isfinite(np.abs(Zin_cpu[:, (f_cpu >= cfg.f_L) & (f_cpu <= cfg.f_H)])).all()

    except Exception:
        traceback.print_exc()
        raise


def main():
    test_pdn_three_caps()
    if matplotlib.get_backend().lower() != "agg":
        img = pathlib.Path(__file__).resolve().parent / "out" / "test_pdn_three_caps.png"
        if img.exists():
            im = plt.imread(img.as_posix())
            plt.figure("preview"); plt.imshow(im); plt.axis("off"); plt.show()


if __name__ == "__main__":
    main()

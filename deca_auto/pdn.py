"""
PDN ラダー合成と Z_in のバッチ計算（GPU ベクトル化）
- sort_indices_by_C_small_to_large(): C 昇順（Load 側 小 → 大）インデックス
- build_static_parasitics(): 周波数上で不変の Z/Y ベクトル群を構築（GPU 常駐）
- zin_batch(): (B, n_freq) の Z_in を一括計算

  * 最終ノード(Load側)は C が小さい。VRM側は C が大きい。
  * ラダー還元は「VRM 側 → Load 側」へ順に：各段で series Z_sN を加え、node で
    Y_c (count×ユニット) を並列合成 → 次段へ伝搬
  * 平面 Y_planar は「最後のコンデンサ段のノード」に並列接続
  * 最後に Z_spread + Z_via を直列で Load まで追加
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from .config import Config, CapSpec


def sort_indices_by_C_small_to_large(C_list: List[float]) -> np.ndarray:
    """
    静電容量 C の昇順インデックス（Load 側 小 → 大）
    - C が None の素子は 0 とみなし、最も小さい側に寄せる（要件: Load 側＝小容量）
    """
    cvals = [c if (c is not None and c > 0) else 0.0 for c in C_list]
    return np.asarray(np.argsort(cvals), dtype=np.int64)


def build_static_parasitics(
    f_dev, cfg: Config, xp, dtype_c
) -> Dict[str, "xp.ndarray"]:
    """
    周波数ベクトル上の寄生 Z/Y を GPU 上で構築
    戻り値キー:
      Z_vrm, Z_stage, Z_spread, Z_via, Y_planar
    """
    w = 2.0 * xp.pi * f_dev

    Z_vrm = xp.asarray(cfg.R_vrm, dtype=dtype_c) + 1j * w * xp.asarray(cfg.L_vrm, dtype=dtype_c)
    Z_stage = xp.asarray(cfg.R_sN, dtype=dtype_c) + 1j * w * xp.asarray(cfg.L_sN, dtype=dtype_c)
    Z_spread = xp.asarray(cfg.R_s, dtype=dtype_c) + 1j * w * xp.asarray(cfg.L_s, dtype=dtype_c)
    Z_via = xp.asarray(cfg.R_v, dtype=dtype_c) + 1j * w * xp.asarray(cfg.L_v, dtype=dtype_c)

    # 平面の損失: Y = jωC + ωC·tanδ、Zp=Rp + 1/Y、Y_planar=1/Zp
    Yc = 1j * w * xp.asarray(cfg.C_p, dtype=dtype_c) + (w * xp.asarray(cfg.C_p, dtype=dtype_c) * xp.asarray(cfg.tan_delta_p, dtype=dtype_c))
    Zp = xp.asarray(cfg.R_p, dtype=dtype_c) + 1.0 / Yc
    Y_planar = 1.0 / Zp

    return {"Z_vrm": Z_vrm, "Z_stage": Z_stage, "Z_spread": Z_spread, "Z_via": Z_via, "Y_planar": Y_planar}


def zin_batch(
    f_dev,
    Zc_list: List["xp.ndarray"],            # 各タイプの Z_c(f)
    counts_dev,                              # shape=(B, n_types)
    parasitics: Dict[str, "xp.ndarray"],
    order_idx: np.ndarray,                   # C 小→大（Load 側の順）
    cap_specs: List[CapSpec] | None,         # L_mnt を取得（Config.validate で補完済）
    xp,
    dtype_c,
):
    """
    まとめて Z_in を計算（GPU）
    - Zc_list は types と同じ順で与えられる想定。order_idx に従い並べ替えて使用。
    - counts_dev: (B, n_types) の非負整数
    戻り値: Z_in_dev: (B, n_freq) complex
    計算法（VRM → Load の順に段を畳み込む）:
      Z_eq を Z_vrm で初期化
      for 各段 k (VRM側: 大容量 → Load側: 小容量):
         Z_eq = Z_eq + Z_stage
         Y_ck_total = count_k * (1 / (jω Lmnt_k + Zc_k))
         Z_eq = 1 / ( 1/Z_eq + Y_ck_total )
      Z_eq = 1 / ( 1/Z_eq + Y_planar )      # 最終ノードに平面
      Z_in = Z_eq + Z_spread + Z_via        # Load までの直列
    """
    B = int(counts_dev.shape[0])
    n_types = len(Zc_list)
    n_freq = int(f_dev.shape[0])

    w = 2.0 * xp.pi * f_dev
    Z_vrm = parasitics["Z_vrm"]
    Z_stage = parasitics["Z_stage"]
    Z_spread = parasitics["Z_spread"]
    Z_via = parasitics["Z_via"]
    Y_planar = parasitics["Y_planar"]

    # 並べ替え：order_idx は C 昇順（Load側 小→大）
    # VRM から畳み込むには「大→小」の順で処理する
    rev = order_idx[::-1]

    # 形状をそろえる（B, F）
    Z_eq = xp.broadcast_to(Z_vrm, (B, n_freq)).astype(dtype_c, copy=False)

    # 段ループ（VRM 側 大容量 → Load 側 小容量）
    for pos, i in enumerate(rev.tolist()):
        Zc_i = Zc_list[i]  # (F,)
        Lmnt_i = xp.asarray(cap_specs[i].L_mnt if cap_specs else 0.0, dtype=dtype_c)

        # series: Z_stage を追加
        Z_eq = Z_eq + Z_stage  # (B,F) + (F,) -> (B,F)

        # シャント: Y_ck_total = count * (1 / (jω Lmnt + Zc))
        Y_unit = 1.0 / (1j * w * Lmnt_i + Zc_i)  # (F,)
        cnt = counts_dev[:, i].reshape(B, 1).astype(dtype_c, copy=False)  # (B,1)
        Y_total = cnt * Y_unit.reshape(1, n_freq)  # (B,F)

        # 並列合成（アドミタンスで）
        Z_eq = 1.0 / (1.0 / Z_eq + Y_total + 1e-30)  # 数値安定化のため微小項

    # 平面 Y_planar を最終ノードに並列
    Z_eq = 1.0 / (1.0 / Z_eq + Y_planar.reshape(1, n_freq) + 1e-30)

    # Load 側直列の拡散＋ビアを追加
    Z_in = Z_eq + Z_spread.reshape(1, n_freq) + Z_via.reshape(1, n_freq)
    return Z_in

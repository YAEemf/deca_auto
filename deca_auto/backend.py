# -*- coding: utf-8 -*-
"""
バックエンド（NumPy/CuPy）の選択と VRAM バジェット・チャンク上限計算
他モジュール互換:
  - select_xp() -> (xp, use_gpu)
  - vram_budget_bytes()
  - decide_max_chunk()
"""
from __future__ import annotations

from typing import Tuple
import math
import os
import traceback

def select_xp(force_numpy: bool = False) -> Tuple[object, bool]:
    """
    CuPy が利用可能かつ force_numpy=False なら CuPy を選択。
    それ以外は NumPy を返す。
    戻り値: (xp_module, use_gpu: bool)
    """
    try:
        if not force_numpy:
            import cupy as cp
            _ = cp.zeros(1)    # デバイス確保で最小動作確認
            free, total = cp.cuda.runtime.memGetInfo()
            print(f"[GPU] Free VRAM: {free/1e6:.1f} MB / Total {total/1e6:.1f} MB")
            return cp, True
    except Exception:
        traceback.print_exc()

    import numpy as np
    return np, False


def vram_budget_bytes(max_vram_ratio: float = 0.9) -> int:
    """
    空き VRAM × max_vram_ratio をバジェットとして返す（GPU 無効なら 0）
    cupy.cuda.runtime.memGetInfo() で free/total を取得
    """
    try:
        import cupy as cp
        free, total = cp.cuda.runtime.memGetInfo()
        budget = int(free * float(max_vram_ratio))
        return max(0, budget)
    except Exception:
        return 0


def decide_max_chunk(n_freq: int, bytes_per_item: int, safety: float = 0.9) -> int:
    """
    VRAM バジェット内で同時評価できる候補数（チャンク上限）を概算
    - n_freq: 周波数点数
    - bytes_per_item: 候補1件あたりの複素ベクトル等の総メモリ（バイト）
      （例）Z_in (n_freq * dtype_c) + 中間バッファ等を見積り
    - safety: 予備マージン
    戻り値: 上限（最低 1）
    """
    budget = vram_budget_bytes()
    if budget <= 0:
        # CPU の場合は「無制限」は危険なので保守的に 64 件
        return 64

    usable = int(budget * float(safety))
    denom = max(1, int(bytes_per_item))
    b = max(1, usable // denom)
    return int(b)

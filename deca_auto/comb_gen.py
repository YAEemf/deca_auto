"""
非負整数解（総数制約あり）の全列挙（CPU）と GPU へのチャンク転送
- enumerate_count_vectors(): ジェネレータ（全ゼロ除外、総数を [ceil(max_total*min_ratio) .. max_total] に制限）
- batch_to_device(): CPU -> GPU 転送（明示的）
他モジュール互換:
  - main.py からジェネレータを受け、pdn.zin_batch() に渡すため xp.ndarray(B, n_types) を作る
"""

from __future__ import annotations

from typing import Iterator, List
import numpy as np
import random
import math
import traceback


def enumerate_count_vectors(
    n_types: int,
    max_total: int,
    min_ratio: float,
    shuffle: bool,
    seed: int | None,
) -> Iterator[np.ndarray]:
    """
    非負整数解（重複組合せ）を全列挙。全ゼロは除外。
    - shuffle=True のとき、探索順のみランダム化（全件は必ず列挙）
      * 総数 t の順を乱択
      * 各レベルの分岐順（0..残り）を乱択
    """
    t_min = max(1, int(np.ceil(max_total * float(min_ratio))))
    totals = list(range(t_min, max_total + 1))
    rng = np.random.default_rng(seed) if shuffle else None
    if shuffle and len(totals) > 1:
        rng.shuffle(totals)

    buf = np.zeros(n_types, dtype=np.int16)

    def _rec(pos: int, remaining: int):
        if pos == n_types - 1:
            buf[pos] = remaining
            yield buf.copy()
            return
        choices = np.arange(remaining + 1, dtype=np.int16)
        if shuffle and choices.size > 1:
            rng.shuffle(choices)  # 分岐順を乱択
        for c in choices:
            buf[pos] = int(c)
            yield from _rec(pos + 1, remaining - int(c))

    for t in totals:
        yield from _rec(0, t)


def batch_to_device(batch_cpu: np.ndarray, xp):
    """
    CPU -> GPU 転送のラッパ
    - Cupy/NumPy 互換: cp.asarray / np.asarray を内部で選択
    """
    try:
        return xp.asarray(batch_cpu)
    except Exception:
        traceback.print_exc()
        # 失敗時はそのまま返す（CPU 計算にフォールバック）
        return batch_cpu

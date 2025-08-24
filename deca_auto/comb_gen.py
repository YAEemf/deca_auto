# -*- coding: utf-8 -*-
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


def _enumerate_fixed_sum(n_types: int, total: int) -> Iterator[np.ndarray]:
    """
    Stars-and-bars を再帰で展開する。
    - n_types 種の非負整数 (x0,...,x_{n-1}) のうち、sum=total をすべて生成
    - 戻り値: shape=(n_types,) の np.int16
    """
    buf = [0] * n_types

    def rec(i: int, rest: int):
        if i == n_types - 1:
            buf[i] = rest
            yield np.asarray(buf, dtype=np.int16)
            return
        # 0..rest まで振り分け
        for v in range(rest + 1):
            buf[i] = v
            yield from rec(i + 1, rest - v)

    yield from rec(0, total)


def enumerate_count_vectors(n_types: int, max_total: int, min_ratio: float,
                            shuffle: bool, seed: int):
    """
    すべての非負整数解（全ゼロ除外、総数制約あり）を列挙。
    shuffle=True の場合でも「全列挙」は保証しつつ順序をランダム化。
    """
    import numpy as _np
    t_min = max(1, int(_np.ceil(max_total * float(min_ratio))))
    totals = list(range(t_min, max_total + 1))
    if shuffle:
        rng = _np.random.default_rng(int(seed))
        rng.shuffle(totals)
    else:
        rng = None

    def _gen_for_total(t, k, prefix):
        """
        t を k 個の非負整数へ分割（残り t、残り次元 k）。
        prefix はこれまでの割り当て。
        shuffle=True のとき、各レベルの候補順（0..t）を乱す。
        """
        if k == 1:
            yield prefix + [t]
            return
        choices = list(range(0, t + 1))
        if rng is not None:
            rng.shuffle(choices)  # 各レベルで順序を乱す
        for x in choices:
            # x をこの次元に割り当て、残り t-x を k-1 次元で分割
            yield from _gen_for_total(t - x, k - 1, prefix + [x])

    for t in totals:
        # 生成されるベクトルは [count_0, ..., count_{n-1}]（元のコンデンサ並びの順）
        for vec in _gen_for_total(t, n_types, []):
            yield _np.asarray(vec, dtype=_np.int16)


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

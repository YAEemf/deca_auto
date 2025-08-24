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


def enumerate_count_vectors(
    n_types: int,
    max_total: int,
    min_ratio: float,
    shuffle: bool,
    seed: int,
) -> Iterator[np.ndarray]:
    """
    全ての非負整数解を列挙（全ゼロ除外）
    - 総数 t を [ceil(max_total*min_ratio) .. max_total] とする
    - shuffle=True の場合、各 t の組合せ順を擬似乱数でシャッフル（探索の多様性向上）
    戻り値: (n_types,) の np.int16 配列を順次 yield
    """
    try:
        t_min = max(1, math.ceil(float(max_total) * float(min_ratio)))
        rng = random.Random(int(seed))

        for t in range(t_min, int(max_total) + 1):
            batch = list(_enumerate_fixed_sum(n_types, t))
            if shuffle:
                rng.shuffle(batch)
            for vec in batch:
                if vec.sum() > 0:
                    yield vec
    except Exception:
        traceback.print_exc()
        return


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

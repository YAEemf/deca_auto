"""
- to_device(a, xp) / to_cpu(a): 明示変換（CuPy/NumPy 互換）
- log_interp1d(x, y, x_new): log-log 線形補間
- trapz_logx(y, x): log(x) での台形積分
- decimate_for_transfer(y, factor): CPU 転送前の間引き
- Timer: with 文で経過時間計測
- get_logger(level): logging 一元化
"""
from __future__ import annotations

from typing import Any, Iterable, Optional
import time
import numpy as np
import logging

def to_device(a, xp):
    """NumPy -> CuPy / CuPy -> CuPy（no-op）を吸収する安全ラッパ"""
    try:
        return xp.asarray(a)
    except Exception:
        return a  # フォールバック（CPU 維持）


def to_cpu(a):
    """CuPy -> NumPy / NumPy -> NumPy（no-op）"""
    try:
        # CuPy 配列は .get() を持つ
        return a.get() if hasattr(a, "get") else np.asarray(a)
    except Exception:
        return a


def log_interp1d(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """log-log 線形補間（端は外挿せず端値保持）"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    lx = np.log10(x); ly = np.log10(y)
    lx_new = np.log10(x_new)
    ly_new = np.interp(lx_new, lx, ly)
    return np.power(10.0, ly_new)


def trapz_logx(y, x):
    """
    log(x) での台形積分（y: (B,F) または (F,)）
    - Matplotlib の autoscale_view のようにデータ範囲から再計算する場合は
      呼び出し側で軸選択を制御する（ここでは純粋計算）
    """
    import numpy as _np
    y = _np.asarray(y, dtype=float)
    x = _np.asarray(x, dtype=float)
    logx = _np.log(x)
    if y.ndim == 1:
        return _np.sum(0.5 * (y[:-1] + y[1:]) * _np.diff(logx))
    else:
        df = _np.diff(logx).reshape(1, -1)
        return _np.sum(0.5 * (y[:, :-1] + y[:, 1:]) * df, axis=1)


def decimate_for_transfer(y, factor: int = 4):
    """CPU 転送前に等間引き（factor>=1）"""
    if factor <= 1:
        return y
    return y[..., ::factor]


class Timer:
    """with 文で経過時間を測る簡易タイマ"""
    def __init__(self, name: str = ""):
        self.name = name
        self.t0 = None
        self.elapsed = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.t0
        if self.name:
            print(f"[Timer] {self.name}: {self.elapsed:.3f}s")


def get_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("deca_auto")
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(level)
    return logger

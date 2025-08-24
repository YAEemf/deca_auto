from __future__ import annotations
from typing import Any
import numpy as np

def analytic_Zc(f_dev, C: float, ESR: float, ESL: float, Lmnt: float, xp, dtype_c):
    """
    R + L + 理想C (+ L_mnt) の解析式による Zc(f)
    Zc = ESR + jω*(ESL + Lmnt)  - j/(ω*C)
    """
    w = 2 * xp.pi * f_dev
    cap = (-1j) / (w * C)

    series_L = 1j * w * (ESL + (Lmnt or 0.0))
    Z = xp.asarray(ESR, dtype=dtype_c) + series_L + cap
    return Z.astype(dtype_c, copy=False)

"""
周波数グリッド生成・評価帯域マスク・目標マスク（折れ線/スカラー）
他モジュール互換:
  - make_freq_grid(cfg, xp) -> f_dev(xp.ndarray)
  - make_eval_band_and_mask(cfg, f_dev, xp) -> (m_eval(bool), z_target(float))
"""
from __future__ import annotations

from typing import Tuple, Optional, List
import numpy as np
import math

from .config import Config

def _loglog_interp(x: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    """
    log-log 線形補間（x, xp, yp はすべて > 0）
    - x: 新しい x（昇順）
    - xp: 既知の x
    - yp: 既知の y
    戻り値: y(x)（線形補間。外挿は端値保持）
    """
    lx = np.log10(x)
    lxp = np.log10(xp)
    lyp = np.log10(yp)
    # numpy.interp は一次元で高速（端は外挿せず端値保持）
    ly = np.interp(lx, lxp, lyp)
    return np.power(10.0, ly)


def make_freq_grid(cfg: Config, xp) -> "xp.ndarray":
    """
    周波数グリッド（logspace）を GPU/CPU 側で生成
    """
    # CuPy/NumPy いずれにも logspace があり挙動は同等
    # start/stop は常用対数
    start = math.log10(cfg.f_start)
    stop = math.log10(cfg.f_stop)
    f_dev = xp.logspace(start, stop, num=int(cfg.num_points), endpoint=True, dtype=xp.float32)
    return f_dev


def _interp_loglog_mask(f_cpu: np.ndarray, mask_xy: np.ndarray) -> np.ndarray:
    """
    折れ線マスクを log–log 直線（べき乗則）で補間して f_cpu 点上の Z_target を返す
    - f_cpu: 1D 周波数（CPUのnp.ndarray、昇順）
    - mask_xy: (K,2) [[f_i, Z_i], ...] 昇順でなくてもよい
    戻り値: 1D Z_target(f_cpu) [Ohm]
    """
    xy = np.asarray(mask_xy, dtype=float)
    xy = xy[np.argsort(xy[:, 0])]                # f 昇順
    fx, zx = xy[:, 0], xy[:, 1]
    logf = np.log10(fx)
    logz = np.log10(zx)
    # f_cpu に対し、区間ごとに log–log 線形補間（外挿は行わない）
    out = np.full_like(f_cpu, np.nan, dtype=float)
    sel = (f_cpu >= fx[0]) & (f_cpu <= fx[-1])
    if np.count_nonzero(sel) == 0:
        return out
    xf = np.log10(f_cpu[sel])
    # np.interp は線形補間なので log 軸に座標変換してから戻す
    logz_f = np.interp(xf, logf, logz)          # logZ の線形補間
    out[sel] = 10.0 ** logz_f
    return out

def make_eval_band_and_mask(cfg, f_dev, xp) -> Tuple["xp.ndarray", "xp.ndarray"]:
    """
    評価帯域マスク m_eval と、f_dev 上の Z_target ベクトルを返す（ともに device 側）
    - カスタムマスクがあれば log–log 直線で補間し、[min_f, max_f] を評価帯域に採用
    - なければ [f_L, f_H] かつ Z_target はフラット
    戻り値:
      m_eval: bool (F,)      device
      z_target: float (F,)   device（評価帯域外は NaN）
    """
    f_cpu = np.asarray(f_dev.get() if hasattr(f_dev, "get") else f_dev, dtype=float)

    if cfg.z_custom_mask:
        zt_cpu = _interp_loglog_mask(f_cpu, np.asarray(cfg.z_custom_mask, dtype=float))
        # 評価帯域はカスタムマスクの範囲
        m_eval_cpu = (~np.isnan(zt_cpu))
        # NaN は比較対象外になるようにしておく
        zt_cpu[~m_eval_cpu] = np.nan
        z_target = xp.asarray(zt_cpu, dtype=getattr(xp, cfg.dtype_r))
        m_eval = xp.asarray(m_eval_cpu, dtype=bool)
        return m_eval, z_target

    # 既定フラットマスク
    m_eval_cpu = (f_cpu >= float(cfg.f_L)) & (f_cpu <= float(cfg.f_H))
    zt_cpu = np.full_like(f_cpu, float(cfg.z_target), dtype=float)
    zt_cpu[~m_eval_cpu] = np.nan   # 帯域外は NaN
    z_target = xp.asarray(zt_cpu, dtype=getattr(xp, cfg.dtype_r))
    m_eval = xp.asarray(m_eval_cpu, dtype=bool)
    return m_eval, z_target
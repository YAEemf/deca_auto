"""
スコア算出・Monte Carlo ロバスト評価・上位抽出（GPU）
- metrics_from_zin(): 指標一式を GPU ベクトル化で算出
- score_linear_comb(): 重み付き合成（小さいほど良い）
- monte_carlo_worst(): 乱数摂動で最悪スコア（必要に応じて RLC 再評価）
- topk_indices(): GPU 上で Top-k を抽出（argpartition→argsort）
注意:
  * CuPy の argpartition は実装上「実質フルソート」で kind/order を未サポート（NumPyと挙動差）。
  * 乱数は xp.random.default_rng(seed)（CuPy は XORWOW）を使用。
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any
import numpy as np

from .config import Config, CapSpec
from . import rlc_model
from . import pdn


def _logspan_and_zref(f_dev, m_eval, z_target, xp) -> Tuple["xp.ndarray", "xp.ndarray"]:
    """帯域の log(f_H/f_L) と zref(帯域内 z_target の中央値) を返す（device スカラー）"""
    f = f_dev[m_eval]
    if f.size >= 2:
        lspan = xp.log(f[-1]) - xp.log(f[0])
        lspan = xp.maximum(lspan, xp.array(1e-9, dtype=f.dtype))
    else:
        lspan = xp.array(1.0, dtype=f.dtype)
    zt = z_target[m_eval]
    zref = xp.nanmedian(zt)
    zref = xp.maximum(zref, xp.array(1e-12, dtype=zt.dtype))
    return lspan, zref


def metrics_from_zin(Zin_dev, f_dev, m_eval, z_target, xp) -> Dict[str, "xp.ndarray"]:
    """
    指標算出（device）。area は log f 台形則。anti は +→− の中心、over のみカウント。
    返すキー: max_over/area_over/under_area/anti_count/anti_height/flat_std + zref/logspan
    """
    sel = m_eval
    Z  = Zin_dev[:, sel]         # (B,Fe)
    zt = z_target[sel]           # (Fe,)
    f  = f_dev[sel]              # (Fe,)

    absZ = xp.abs(Z)
    logZ = xp.log10(absZ + 1e-30)

    over  = xp.maximum(absZ - zt[None, :], 0.0)
    under = xp.maximum(zt[None, :] - absZ, 0.0)

    max_over = xp.nanmax(over, axis=1)

    lfx = xp.log(f)
    if f.size >= 2:
        dlf = xp.diff(lfx)                                  # (Fe-1,)
        area_over  = xp.sum(0.5 * (over[:, :-1]  + over[:, 1:])  * dlf[None, :], axis=1)
        area_under = xp.sum(0.5 * (under[:, :-1] + under[:, 1:]) * dlf[None, :], axis=1)
    else:
        area_over  = xp.zeros((Z.shape[0],), dtype=absZ.dtype)
        area_under = xp.zeros((Z.shape[0],), dtype=absZ.dtype)

    sgn   = xp.sign(xp.diff(logZ, axis=1))
    peaks = (sgn[:, :-1] > 0) & (sgn[:, 1:] < 0)
    over_center = (over > 0.0)[:, 1:-1]
    peaks = peaks & over_center
    anti_count  = peaks.sum(axis=1)
    anti_height = xp.where(peaks, over[:, 1:-1], 0.0).max(axis=1) if peaks.shape[1] > 0 else xp.zeros_like(max_over)

    flat_std = xp.nanstd(logZ, axis=1)

    # 正規化用スカラー（device）
    logspan, zref = _logspan_and_zref(f_dev, m_eval, z_target, xp)

    out_dtype = absZ.dtype
    return {
        "max_over":   max_over.astype(out_dtype, copy=False),
        "area_over":  area_over.astype(out_dtype, copy=False),
        "under_area": area_under.astype(out_dtype, copy=False),
        "anti_count": anti_count.astype(out_dtype, copy=False),
        "anti_height": anti_height.astype(out_dtype, copy=False),
        "flat_std":   flat_std.astype(out_dtype, copy=False),
        # 正規化用
        "zref":       zref.astype(out_dtype, copy=False),
        "logspan":    logspan.astype(out_dtype, copy=False),
    }


def topk_indices(scores_dev, k: int, xp):
    """下位kの安定抽出。NaN/±Inf を無害化し float64 で比較。"""
    if k <= 0:
        return xp.asarray([], dtype=int)
    s = xp.nan_to_num(scores_dev, posinf=1e30, neginf=-1e30).astype(xp.float64, copy=False)
    n = int(s.shape[0]); k = min(int(k), n)
    if k == 0:
        return xp.asarray([], dtype=int)
    frac = k / max(1, n)
    if k <= 256 and frac <= 0.05:
        return xp.argsort(s)[:k]
    idx = xp.argpartition(s, kth=k - 1)[:k]
    return idx[xp.argsort(s[idx])]


def score_linear_comb(metrics: Dict[str, "xp.ndarray"], w: dict, parts_count, xp):
    """
    合成スコア（小さいほど良い）。無次元化し、float64 で加算して比較精度を確保。
    """
    # 正規化係数（device スカラー）
    zref    = metrics.get("zref",    None)
    logspan = metrics.get("logspan", None)
    if (zref is None) or (logspan is None):
        # 後方互換フォールバック（通常は到達しない）
        zref    = xp.maximum(xp.median(metrics["max_over"]), xp.array(1e-12, dtype=metrics["max_over"].dtype))
        logspan = xp.array(1.0, dtype=zref.dtype)

    zref64    = zref.astype(xp.float64, copy=False)
    logspan64 = logspan.astype(xp.float64, copy=False)

    # 無次元化（float64）
    max_over_n   = (metrics["max_over"].astype(xp.float64)   / zref64)
    area_over_n  = (metrics["area_over"].astype(xp.float64)  / (zref64 * logspan64))
    under_area_n = (metrics["under_area"].astype(xp.float64) / (zref64 * logspan64))
    anti_h_n     = (metrics["anti_height"].astype(xp.float64)/ zref64)
    flat_std     =  metrics["flat_std"].astype(xp.float64)

    s = xp.zeros_like(max_over_n, dtype=xp.float64)
    s += float(w.get("max",  0.0)) * max_over_n
    s += float(w.get("area", 0.0)) * area_over_n
    s += float(w.get("flat", 0.0)) * flat_std
    s += float(w.get("under",0.0)) * under_area_n
    s += float(w.get("anti", 0.0)) * (metrics["anti_count"].astype(xp.float64) + 0.25 * anti_h_n)
    if w.get("parts", 0.0) != 0.0:
        s += float(w["parts"]) * parts_count.astype(xp.float64)
    return s  # float64 device


def topk_indices(scores_dev, k: int, xp):
    """
    下位kの安定抽出。
    - kが小さい/比率が小さいときは argsort で安定抽出
    - それ以外は argpartition→argsort
    - NaN/±Inf は安全な有限値に置換
    """
    if k <= 0:
        return xp.asarray([], dtype=int)
    s = xp.nan_to_num(scores_dev, posinf=1e30, neginf=-1e30)
    n = int(s.shape[0]); k = min(int(k), n)
    if k == 0:
        return xp.asarray([], dtype=int)

    # 比較精度を上げるため float64 に一時変換（GPUでも可）
    s64 = s.astype(xp.float64, copy=False)
    frac = k / max(1, n)
    if k <= 256 and frac <= 0.05:
        idx = xp.argsort(s64)[:k]
        return idx
    # 部分抽出→部分を安定並べ替え
    idx = xp.argpartition(s64, kth=k - 1)[:k]
    idx = idx[xp.argsort(s64[idx])]
    return idx


# ---------------- Monte Carlo ロバスト評価 ----------------
def _perturb_RL_only(xp, Zc_unit, f_dev, tol_ESR: float, tol_ESL: float, rng):
    """VF モデル等で C が未知の場合の簡易摂動: 直列 ΔR, ΔL のみ"""
    B = Zc_unit.reshape(1, -1).astype(Zc_unit.dtype, copy=False)
    # ΔR, ΔL を相対（±tol）で与える（平均0、均一分布）
    dR = (rng.random() * 2.0 - 1.0) * tol_ESR
    dL = (rng.random() * 2.0 - 1.0) * tol_ESL
    w = 2.0 * xp.pi * f_dev
    dZ = xp.asarray(dR, dtype=Zc_unit.dtype) + 1j * w * xp.asarray(dL, dtype=Zc_unit.dtype)
    return Zc_unit + dZ  # (F,)


def monte_carlo_worst(
    Zc_list_base: List["xp.ndarray"],
    counts_dev,                        # (B, n_types)
    cfg: Config,
    f_dev,
    parasitics: Dict[str, "xp.ndarray"],
    order_idx: np.ndarray,
    cap_specs: List[CapSpec] | None,
    xp,
    dtype_c,
):
    """
    Monte Carlo による最悪スコア（小さいほど良い）を返す
    - 乱数: xp.random.default_rng(cfg.seed)（CuPy は XORWOW）
    - C/ESR/ESL に相対公差、mlcc_derating を適用
    - C が既知 (cap.C is not None) の場合: 解析式で再合成（厳密）
      それ以外（VF モデル等）  : 直列 ΔR, ΔL のみ摂動（近似）
    戻り値: worst: (B,)  各候補の MC 最悪スコア
    """

    rng = xp.random.default_rng(int(cfg.seed)) 
    B = int(counts_dev.shape[0])
    n_types = len(Zc_list_base)

    # 評価用に目標/帯域を既に CPU から受けている前提（metrics_from_zin を再利用）
    from .freqgrid import make_eval_band_and_mask
    m_eval, z_target = make_eval_band_and_mask(cfg, f_dev, xp)

    worst = xp.full((B,), -xp.inf, dtype=z_target.dtype)

    for s in range(int(cfg.mc_samples)):
        # 各種類ごとに摂動 Zc を作成
        Zc_list_mc: List["xp.ndarray"] = []
        for i in range(n_types):
            if cap_specs and (cap_specs[i].C is not None):
                # 解析式で再合成
                # 相対公差（均一分布 ±tol）、MLCC デレーティング（容量を低下）
                dC = (rng.random() * 2.0 - 1.0) * float(cfg.tol_C)
                dR = (rng.random() * 2.0 - 1.0) * float(cfg.tol_ESR)
                dL = (rng.random() * 2.0 - 1.0) * float(cfg.tol_ESL)
                C_eff = (1.0 - float(cfg.mlcc_derating)) * cap_specs[i].C * (1.0 + dC)
                ESR_eff = cap_specs[i].ESR * (1.0 + dR)
                ESL_eff = cap_specs[i].ESL * (1.0 + dL)
                Lmnt_eff = (cap_specs[i].L_mnt or cfg.L_mntN) * (1.0 + dL)
                Zc_i = rlc_model.analytic_Zc(
                    f_dev=f_dev, C=C_eff, ESR=ESR_eff, ESL=ESL_eff, Lmnt=0.0, xp=xp, dtype_c=dtype_c
                )
            else:
                # VF モデル等：直列 ΔR, ΔL のみ
                Zc_i = _perturb_RL_only(xp, Zc_list_base[i], f_dev, cfg.tol_ESR, cfg.tol_ESL, rng)
            Zc_list_mc.append(Zc_i)

        # Z_in を再評価
        Zin = pdn.zin_batch(
            f_dev=f_dev,
            Zc_list=Zc_list_mc,
            counts_dev=counts_dev,
            parasitics=parasitics,
            order_idx=order_idx,
            cap_specs=cap_specs,
            xp=xp,
            dtype_c=dtype_c,
        )

        # スコア化（重みは mc_worst のみ使うため、max_over を代表値として採用）
        mets = metrics_from_zin(Zin, f_dev, m_eval, z_target, xp)
        # ここでは「最大超過」を最悪度として使用（加重は上位で行う）
        worst = xp.maximum(worst, mets["max_over"])

    return xp.nan_to_num(worst)

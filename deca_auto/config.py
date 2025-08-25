# -*- coding: utf-8 -*-
# deca_auto/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import math
import copy
import sys
import traceback

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None  # ユーザー環境に合わせる

from .utils import get_logger


@dataclass
class CapSpec:
    name: str
    model_path: Optional[str] = None
    C: Optional[float] = None
    ESR: float = 20e-3
    ESL: float = 1e-9
    L_mnt: Optional[float] = None  # 未指定時は L_mntN を使用


@dataclass
class Config:
    # 周波数グリッド
    f_start: float = 1e2
    f_stop: float = 1e9
    num_points: int = 768

    # 評価帯域・目標
    f_L: float = 1e3
    f_H: float = 1e8
    z_target: float = 10e-3
    z_custom_mask: Optional[List[Tuple[float, float]]] = None  # [[f, Z], ...]

    # PDN 寄生
    R_vrm: float = 15e-3
    L_vrm: float = 10e-9
    R_sN: float = 0.25e-3
    L_sN: float = 1e-9
    L_mntN: float = 1e-9
    R_s: float = 0.1e-3
    L_s: float = 1e-9
    R_v: float = 0.1e-3
    L_v: float = 1e-9
    R_p: float = 1e-3
    C_p: float = 10e-12
    tan_delta_p: float = 0.02

    # Spice
    dc_bias: float = 3.3

    # コンデンサ
    capacitors: List[CapSpec] = field(default_factory=list)

    # 組合せ探索
    max_total_parts: int = 12
    min_total_parts_ratio: float = 0.3
    top_k: int = 10
    shuffle_evaluation: bool = True

    # スコア重み
    weights: Dict[str, float] = field(default_factory=lambda: {
        "max": 0.9,
        "area": 0.65,
        "anti": 0.35,
        "flat": 0.15,
        "under": -0.2,
        "parts": 0.0,
        "mc_worst": 1.0,
    })

    # Monte Carlo
    mc_enable: bool = True
    mc_samples: int = 64
    tol_C: float = 0.2
    tol_ESR: float = 0.2
    tol_ESL: float = 0.2
    mlcc_derating: float = 0.15

    # 乱数・デバイス
    seed: int = 1234
    max_vram_ratio: float = 0.9
    dtype_c: str = "complex64"
    dtype_r: str = "float32"
    force_numpy: bool = False

    # 表示・Excel
    plot_view: bool = True
    plot_min_interval_s: float = 0.5
    plot_force_update_interval_s: float = 5.0
    excel_path: Optional[str] = None
    excel_name: str = "dcap_result"

_DEFAULT = Config()  # 参照用（変更しない）


_TOPLEVEL_KEYS = {
    "f_start", "f_stop", "num_points",
    "f_L", "f_H", "z_target", "z_custom_mask",
    "R_vrm", "L_vrm", "R_sN", "L_sN", "L_mntN", "R_s", "L_s", "R_v", "L_v",
    "R_p", "C_p", "tan_delta_p",
    "dc_bias",
    "max_total_parts", "min_total_parts_ratio", "top_k", "shuffle_evaluation",
    "weights",  # weights は table
    "mc_enable", "mc_samples", "tol_C", "tol_ESR", "tol_ESL", "mlcc_derating",
    "seed", "max_vram_ratio", "dtype_c", "dtype_r", "force_numpy",
    "plot_view", "plot_min_interval_s", "plot_force_update_interval_s",
    "excel_path", "excel_name",
    "capacitors",  # AOT
}

_WEIGHT_KEYS = {"max", "area", "anti", "flat", "under", "parts", "mc_worst"}

def _deep_merge(dst: dict, src: dict) -> dict:
    """
    dict の部分上書きマージ。
    - src にあるキーだけ dst に反映（既定の保持）
    - ネストは再帰。ただし list は置換（AOT など）
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst

def _as_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return x

def _as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "on")

def _ensure_list_tuple_pairs(v) -> Optional[List[Tuple[float, float]]]:
    if v is None:
        return None
    out = []
    for p in v:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            out.append((float(p[0]), float(p[1])))
    return out or None

def _san_cap(cap: dict, l_mntN_default: float) -> CapSpec:
    # 許容キーだけ取り出し
    c = CapSpec(
        name=str(cap.get("name")),
        model_path=cap.get("model_path"),
        C=_as_float(cap.get("C")) if cap.get("C") is not None else None,
        ESR=_as_float(cap.get("ESR")) if cap.get("ESR") is not None else 20e-3,
        ESL=_as_float(cap.get("ESL")) if cap.get("ESL") is not None else 1e-9,
        L_mnt=_as_float(cap.get("L_mnt")) if cap.get("L_mnt") is not None else l_mntN_default,
    )
    if not c.name:
        raise ValueError("capacitors[].name は必須です")
    return c


def _load_toml_as_dict(path: Optional[str]) -> dict:
    if not path:
        return {}
    if tomllib is None:
        raise RuntimeError("tomllib が利用できません。Python 3.11+ が必要です。")
    with open(path, "rb") as f:
        return tomllib.load(f)

def _hoist_orphan_keys(cfg_raw: dict, logger) -> dict:
    """
    誤って [[capacitors]] や [weights] の“中に”書かれたトップレベル用キーを
    トップレベルへ移してから返す。
    - 既にトップレベルに同名キーがあればトップレベルを優先し、
      セクション内の値は無視して削除。
    """
    out = copy.deepcopy(cfg_raw)

    if isinstance(out.get("capacitors"), list):
        fixed_caps = []
        for idx, cap in enumerate(out["capacitors"]):
            if not isinstance(cap, dict):
                continue
            cap = cap.copy()
            stray = []
            for k in list(cap.keys()):
                if k in _TOPLEVEL_KEYS and k not in {"capacitors", "weights"}:
                    # 迷い込んだトップレベル用キー
                    stray.append((k, cap.pop(k)))
            if stray:
                for k, v in stray:
                    if k not in out:  # トップレベルに未定義なら救出
                        out[k] = v
                        logger.warning(f"[config] [[capacitors]] 内の '{k}' をトップレベルへ移動しました。")
                    else:
                        logger.warning(f"[config] [[capacitors]] 内の '{k}' を無視（トップレベル優先）。")
            fixed_caps.append(cap)
        out["capacitors"] = fixed_caps

    if isinstance(out.get("weights"), dict):
        w = out["weights"].copy()
        for k in list(w.keys()):
            if k in _TOPLEVEL_KEYS and k not in _WEIGHT_KEYS:
                v = w.pop(k)
                if k not in out:
                    out[k] = v
                    logger.warning(f"[config] [weights] 内の '{k}' をトップレベルへ移動しました。")
                else:
                    logger.warning(f"[config] [weights] 内の '{k}' を無視（トップレベル優先）。")
        out["weights"] = w

    return out


def load_user_config(toml_path: Optional[str]) -> dict:
    """
    TOML を読み込み、**迷い込んだトップレベル用キーを救出**した上で dict を返す。
    返す dict は **ユーザー指定分のみ**（既定は含めない）。
    """
    logger = get_logger()
    try:
        raw = _load_toml_as_dict(toml_path)
        if not raw:
            return {}
        raw = _hoist_orphan_keys(raw, logger)
        return raw
    except Exception:
        traceback.print_exc()
        # エラー時は空 dict（既定で動作させる）
        return {}


def validate_config(raw: dict) -> Config:
    """
    既定 Config に raw を **ディープマージ**して最終 Config を返す。
    - 未指定は既定を保持
    - list は置換（capacitors）
    - 値の型を最低限バリデート
    """
    logger = get_logger()
    base = copy.deepcopy(_DEFAULT).__dict__
    user = copy.deepcopy(raw)

    # [[capacitors]]: list でなければ無視
    caps_raw = user.get("capacitors")
    if caps_raw is not None and not isinstance(caps_raw, list):
        logger.warning("[config] 'capacitors' は配列（[[capacitors]])で指定してください。無視します。")
        user.pop("capacitors", None)

    # [weights]: dict 部分更新に対応（深いマージ）
    if "weights" in user and not isinstance(user["weights"], dict):
        logger.warning("[config] 'weights' はテーブル（[weights])で指定してください。無視します。")
        user.pop("weights", None)

    # ディープマージ（list は置換）
    merged = _deep_merge(base, user)

    # ---- 型整形・下限上限 ----
    # z_custom_mask
    merged["z_custom_mask"] = _ensure_list_tuple_pairs(merged.get("z_custom_mask"))

    # bool / float / int
    merged["shuffle_evaluation"] = _as_bool(merged.get("shuffle_evaluation"))
    merged["mc_enable"] = _as_bool(merged.get("mc_enable"))
    merged["force_numpy"] = _as_bool(merged.get("force_numpy"))
    for k in ("f_start", "f_stop", "f_L", "f_H", "z_target", "dc_bias", "max_vram_ratio",
              "plot_min_interval_s", "plot_force_update_interval_s"):
        merged[k] = _as_float(merged.get(k))
    for k in ("num_points", "max_total_parts", "top_k", "mc_samples", "seed"):
        try:
            merged[k] = int(merged.get(k))
        except Exception:
            pass
    for k in ("min_total_parts_ratio", "tol_C", "tol_ESR", "tol_ESL", "mlcc_derating"):
        merged[k] = _as_float(merged.get(k))

    # 範囲チェック
    merged["num_points"] = max(4, int(merged["num_points"]))
    merged["min_total_parts_ratio"] = float(min(1.0, max(0.0, merged["min_total_parts_ratio"])))
    merged["max_total_parts"] = max(1, int(merged["max_total_parts"]))
    merged["top_k"] = max(1, int(merged["top_k"]))
    merged["max_vram_ratio"] = float(min(0.99, max(0.1, merged["max_vram_ratio"])))

    # dtype
    if merged.get("dtype_c") not in ("complex64", "complex128"):
        logger.warning("[config] dtype_c は 'complex64' か 'complex128' を推奨します。既定を使用します。")
        merged["dtype_c"] = _DEFAULT.dtype_c
    if merged.get("dtype_r") not in ("float32", "float64"):
        logger.warning("[config] dtype_r は 'float32' か 'float64' を推奨します。既定を使用します。")
        merged["dtype_r"] = _DEFAULT.dtype_r

    # weights: 未知キーは警告して無視
    w = merged.get("weights", {})
    if w:
        w_clean = {}
        for k, v in w.items():
            if k in _WEIGHT_KEYS:
                w_clean[k] = float(v)
            else:
                logger.warning(f"[config] [weights].{k} は未知キーのため無視します。")
        # 欠損は既定を残す
        merged["weights"] = {**_DEFAULT.weights, **w_clean}
    else:
        merged["weights"] = copy.deepcopy(_DEFAULT.weights)

    # capacitors: 正規化
    caps_out: List[CapSpec] = []
    if isinstance(caps_raw, list):
        for i, c in enumerate(caps_raw):
            try:
                caps_out.append(_san_cap(c, merged["L_mntN"]))
            except Exception as e:
                logger.warning(f"[config] capacitors[{i}] を無視します: {e}")
    merged["capacitors"] = caps_out

    # 依存関係チェック：評価帯域
    if merged["z_custom_mask"] is not None:
        # カスタムマスクの min/max が評価帯域を上書きする仕様
        fs = [p[0] for p in merged["z_custom_mask"]]
        fmin, fmax = min(fs), max(fs)
        if fmax <= fmin:
            logger.warning("[config] z_custom_mask の周波数が不正です。既定の評価帯域を使用します。")
        else:
            merged["f_L"], merged["f_H"] = float(fmin), float(fmax)

    # コンビ枚数の下限（仕様どおり ceil）
    t_min = max(1, math.ceil(merged["max_total_parts"] * merged["min_total_parts_ratio"]))
    merged["_t_min"] = int(t_min)  # comb_gen 側仕様の明示（デバッグ用）

    # dataclass へ
    try:
        cfg = Config(**merged)
    except TypeError:
        # 予期せぬキーが残っても落とさないよう、許容キーだけで生成
        allowed = {f.name for f in Config.__dataclass_fields__.values()}
        safed = {k: v for k, v in merged.items() if k in allowed}
        cfg = Config(**safed)

    return cfg

"""
非ブロッキング描画の管理
- init_figures(plot_view: bool) -> dict
- show_loading(ax1)
- plot_caps_Zc(ax1, f_cpu, Zc_cpu_dict)
- live_update_topk(ax2, f_cpu, topk_list, eval_lines, now_ts, min_interval, force_interval)
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import time
import numpy as np
import matplotlib.pyplot as plt


def init_figures(plot_view: bool):
    import matplotlib.pyplot as plt
    figs = {"fig1": None, "ax1": None, "fig2": None, "ax2": None}
    if not plot_view:
        return figs
    plt.ion()
    fig1, ax1 = plt.subplots(num="Decap Zc models")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.grid(True, which="both", alpha=0.4)
    ax1.set_xlabel("Frequency [Hz]"); ax1.set_ylabel("|Zc| [Ohm]")
    figs["fig1"], figs["ax1"] = fig1, ax1
    fig1.canvas.draw(); fig1.canvas.flush_events()
    return figs


def show_loading(ax1):
    """図1へ「Loading capacitor models...」を表示"""
    if ax1 is None:
        return
    ax1.text(0.5, 0.5, "Loading capacitor models...", ha="center", va="center", transform=ax1.transAxes)
    ax1.figure.canvas.draw(); ax1.figure.canvas.flush_events()


def plot_caps_Zc(ax1, f_cpu: np.ndarray, Zc_cpu_dict: Dict[str, np.ndarray]):
    """
    各コンデンサの |Zc| を両対数で1度だけ描画
    - Zc_cpu_dict: {name: Zc(f)_cpu}
    """
    if ax1 is None:
        return
    ax1.cla()
    ax1.grid(True, which="both", alpha=0.4)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Frequency [Hz]"); ax1.set_ylabel("|Zc| [Ohm]")

    for name, Zc in Zc_cpu_dict.items():
        if Zc is None or len(Zc) == 0:
            continue
        ax1.plot(f_cpu, np.abs(Zc), lw=1.2, alpha=0.9, label=name)

    if ax1.has_data():
        ax1.legend(loc="upper left", fontsize=8, framealpha=0.5)
        ax1.set_xlim(float(f_cpu[0]), float(f_cpu[-1]))
        ax1.relim(); ax1.autoscale_view(scalex=False, scaley=True)  # Y オート  :contentReference[oaicite:3]{index=3}

    ax1.figure.canvas.draw(); ax1.figure.canvas.flush_events()


def prepare_fig2(ax2, f_cpu, eval_lines, message: str = "Searching in progress..."):
    """
    グラフ2のウィンドウを準備して、プレースホルダのテキストを表示する
    戻り値: (timestamp, ax2)  ※ live_update_topk と同じタプル形式で返す
    """
    import time as _time
    import matplotlib.pyplot as plt

    if ax2 is None:
        fig2, ax2 = plt.subplots(num="Global Top-k |Zin| (live)")
    else:
        fig2 = ax2.figure

    ax2.cla()
    ax2.grid(True, which="both", alpha=0.4)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("Frequency [Hz]"); ax2.set_ylabel("|Zin| [Ohm]")
    ax2.set_xlim(float(f_cpu[0]), float(f_cpu[-1]))

    # 目標マスク・評価帯域の補助線
    if eval_lines:
        if "band" in eval_lines and eval_lines["band"]:
            fL, fH = eval_lines["band"]
            ax2.axvline(fL, ls="--", c="k", lw=1.0, alpha=0.6)
            ax2.axvline(fH, ls="--", c="k", lw=1.0, alpha=0.6)
        if eval_lines.get("mask_x") is not None and eval_lines.get("mask_y") is not None:
            ax2.plot(eval_lines["mask_x"], eval_lines["mask_y"], "--", c="k", lw=1.0, alpha=0.8, label="Target mask")

    # 中央にテキスト
    ax2.text(0.5, 0.5, message, transform=ax2.transAxes, ha="center", va="center", fontsize=11, alpha=0.8)
    if ax2.has_data():
        ax2.legend(loc="upper left", fontsize=8, framealpha=0.5)

    ax2.relim(); ax2.autoscale_view(scalex=False, scaley=True)
    fig2.canvas.draw(); fig2.canvas.flush_events()
    ts = _time.time()
    ax2._last_update_ts = ts
    return ts, ax2


def live_update_topk(
    ax2,
    f_cpu: np.ndarray,
    topk_list: List[Tuple[str, np.ndarray]],  # ここは「スコア昇順」で渡す（最優秀が先頭）
    eval_lines: Dict,
    now_ts: float,
    min_interval: float,
    force_interval: float,
):
    """
    グローバル Top-k の |Zin| をライブ描画（非ブロッキング）
    - 初回は必ず描画
    - 2回目以降は min_interval/force_interval に従って更新タイミング調整
    - 凡例は上から「スコア昇順（優秀）」、"Target mask" は一番下
    """
    import time as _time
    import matplotlib.pyplot as plt

    if ax2 is None:
        fig2, ax2 = plt.subplots(num="Global Top-k |Zin| (live)")
    else:
        fig2 = ax2.figure

    # 時刻補正
    if now_ts is None or now_ts <= 0.0:
        now_ts = _time.time()

    last = getattr(ax2, "_last_update_ts", 0.0)
    first_draw = (last is None) or (last <= 0.0)
    if (not first_draw) and ((now_ts - last) < float(min_interval)):
        if (now_ts - last) < float(force_interval):
            return last, ax2

    ax2.cla()
    ax2.grid(True, which="both", alpha=0.4)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("Frequency [Hz]"); ax2.set_ylabel("|Zin| [Ohm]")
    ax2.set_xlim(float(f_cpu[0]), float(f_cpu[-1]))

    # 目標マスク・評価帯域（先にプロットして常に背面へ）
    mask_handle = None
    if eval_lines:
        if "band" in eval_lines and eval_lines["band"]:
            fL, fH = eval_lines["band"]
            ax2.axvline(fL, ls="--", c="k", lw=1.0, alpha=0.6, zorder=1)
            ax2.axvline(fH, ls="--", c="k", lw=1.0, alpha=0.6, zorder=1)
        if eval_lines.get("mask_x") is not None and eval_lines.get("mask_y") is not None:
            # マスク線は zorder 低め & 凡例で最後に回す
            (mask_handle,) = ax2.plot(
                eval_lines["mask_x"], eval_lines["mask_y"],
                "--", c="k", lw=1.0, alpha=0.8, label="Target mask", zorder=1
            )

    # 上位トレース（スコア昇順で渡される前提）
    # 最優秀を「前面（上）」にするため、昇順でそのまま描く（最後に描いたものが最前面）
    handle_map = {}
    for label, Zin in topk_list:
        if Zin is None:
            continue
        (h,) = ax2.plot(f_cpu, np.abs(Zin), lw=1.2, alpha=0.95, label=label, zorder=2)
        handle_map[label] = h

    #   1) Top_k のラベル（= スコア昇順）を上から
    #   2) "Target mask" を一番下に追加
    if handle_map or mask_handle is not None:
        desired_labels = list(handle_map.keys())  # ここは topk_list の順に一致
        desired_handles = [handle_map[lbl] for lbl in desired_labels]
        if mask_handle is not None:
            desired_labels.append("Target mask")
            desired_handles.append(mask_handle)

        ax2.legend(desired_handles, desired_labels, loc="upper left", fontsize=8, framealpha=0.5)

    # 非ブロッキング更新
    ax2.relim(); ax2.autoscale_view(scalex=False, scaley=True)
    fig2.canvas.draw(); fig2.canvas.flush_events()
    ts = _time.time()
    ax2._last_update_ts = ts
    return ts, ax2

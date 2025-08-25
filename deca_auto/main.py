from __future__ import annotations

import argparse
import sys
import math
import time
import traceback
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm

from . import config as cfgmod
from .backend import select_xp, decide_max_chunk
from .freqgrid import make_freq_grid, make_eval_band_and_mask
from .spice_model import build_capacitor_model
from .comb_gen import enumerate_count_vectors, batch_to_device
from .pdn import build_static_parasitics, sort_indices_by_C_small_to_large, zin_batch
from .score import metrics_from_zin, score_linear_comb, topk_indices, monte_carlo_worst
from .plotter import init_figures, show_loading, plot_caps_Zc, live_update_topk, prepare_fig2
from .excel_out import write_topk_result
from .utils import get_logger, to_cpu, decimate_for_transfer


def _estimate_bytes_per_item(n_freq: int, dtype_c_bytes: int, dtype_r_bytes: int, safety: float = 2.0) -> int:
    per_f = (2 * dtype_c_bytes) + (4 * dtype_r_bytes)
    return int(n_freq * per_f * float(safety))


def _estimate_total_combos(n_types: int, max_total: int, min_ratio: float) -> int:
    t_min = max(1, math.ceil(max_total * float(min_ratio)))
    total = 0
    for t in range(t_min, max_total + 1):
        total += math.comb(t + n_types - 1, n_types - 1)
    return total


def _make_target_polyline(cfg: cfgmod.Config) -> np.ndarray | None:
    if cfg.z_custom_mask:
        return np.asarray(cfg.z_custom_mask, dtype=float)
    return np.asarray([[cfg.f_L, cfg.z_target], [cfg.f_H, cfg.z_target]], dtype=float)


def main(toml_path: str | None = None):
    logger = get_logger()

    try:
        # 1 設定
        raw = cfgmod.load_user_config(toml_path)
        cfg = cfgmod.validate_config(raw)
        print(cfg)

        # 2 バックエンド
        xp, use_gpu = select_xp(force_numpy=cfg.force_numpy)
        logger.info(f"Backend: {'CuPy/GPU' if use_gpu else 'NumPy/CPU'}")

        # 3 グリッド・評価帯域
        f_dev = make_freq_grid(cfg, xp)
        m_eval, z_target = make_eval_band_and_mask(cfg, f_dev, xp)
        f_cpu = to_cpu(f_dev)
        target_poly = _make_target_polyline(cfg)

        # 4 図1のみ先に作成
        figs = init_figures(cfg.plot_view)
        if cfg.plot_view and figs["ax1"] is not None:
            show_loading(figs["ax1"])

        # 5 Zc 事前計算
        Zc_list_dev: List["xp.ndarray"] = []
        C_list = [c.C for c in cfg.capacitors]
        for cap in tqdm(cfg.capacitors, desc="Cap models", unit="cap",
                        dynamic_ncols=True, mininterval=0.2):
            Zc = build_capacitor_model(cap, f_dev, xp)
            Zc_list_dev.append(Zc)

        if cfg.plot_view and figs["ax1"] is not None:
            Zc_cpu_dict = {cap.name: to_cpu(z) for cap, z in zip(cfg.capacitors, Zc_list_dev)}
            plot_caps_Zc(figs["ax1"], f_cpu, Zc_cpu_dict)

        # 6 段順・寄生
        order_idx = sort_indices_by_C_small_to_large(C_list)
        parasitics = build_static_parasitics(f_dev, cfg, xp, getattr(xp, cfg.dtype_c))

        # 7 チャンク上限・フラッシュ閾値
        n_freq = int(f_dev.shape[0])
        dtype_c_bytes = (8 if cfg.dtype_c == "complex64" else 16)
        dtype_r_bytes = (4 if cfg.dtype_r == "float32" else 8)
        bytes_per_item = _estimate_bytes_per_item(n_freq, dtype_c_bytes, dtype_r_bytes, safety=2.0)
        max_chunk = decide_max_chunk(n_freq, bytes_per_item)
        logger.info(f"Chunk limit (estimated): {max_chunk}")
        flush_threshold = min(max_chunk, 1024*16)

        # 8 プログレス（総数は理論値）
        if len(cfg.capacitors) == 0:
            logger.warning("コンデンサの候補がありません")
            sys.exit(1)
        total_est = _estimate_total_combos(len(cfg.capacitors), cfg.max_total_parts, cfg.min_total_parts_ratio)
        pbar = tqdm(total=total_est, desc="Enumerating combos", unit="combos",
                    dynamic_ncols=True, mininterval=0.2)

        # 9 グラフ2を探索開始の直前にプレースホルダで表示
        eval_lines = {
            "band": (cfg.f_L, cfg.f_H),
            "mask_x": target_poly[:, 0] if target_poly is not None else None,
            "mask_y": target_poly[:, 1] if target_poly is not None else None,
        }
        last_plot_ts = 0.0
        if cfg.plot_view:
            last_plot_ts, ax2 = prepare_fig2(figs.get("ax2"), f_cpu, eval_lines, "Searching in progress...")
            figs["ax2"] = ax2
            figs["fig2"] = None if ax2 is None else ax2.figure

        # 10 探索ループ
        global_best_scores: np.ndarray | None = None
        global_best_records: List[Dict] = []
        global_best_traces: List[Tuple[str, np.ndarray]] = []
        dtype_c = getattr(xp, cfg.dtype_c)

        global_candidates = {}  # key: tuple(counts in order_idx), value: dict(label, score, Zin_cpu, total)

        def _merge_topk_and_plot(scores_dev, Zin_dev, counts_dev_chunk):
            nonlocal global_candidates, figs, last_plot_ts, global_best_scores, global_best_records, global_best_traces

            # デバイス→CPU（小さいデータのみ）
            top_idx_local = topk_indices(scores_dev, cfg.top_k, xp)
            idx_cpu = to_cpu(top_idx_local)
            top_scores_cpu = to_cpu(scores_dev[top_idx_local]).astype(float)
            top_z_cpu = to_cpu(Zin_dev[top_idx_local])
            top_counts_cpu = to_cpu(counts_dev_chunk[idx_cpu])

            # チャンク内Top-kをグローバル辞書にマージ（重複は良い方で上書き）
            for i in range(len(idx_cpu)):
                cnt = tuple(int(v) for v in top_counts_cpu[i].tolist())
                total = int(sum(cnt))
                # ラベルは C小→大の順で "(name)xN + ..."（Total:n）
                parts = []
                for j in order_idx.tolist():
                    if cnt[j] > 0:
                        parts.append(f"({cfg.capacitors[j].name})x{cnt[j]}")
                label = " + ".join(parts) + f" (Total:{total})" if parts else "(none)"

                score = float(top_scores_cpu[i])
                if (cnt not in global_candidates) or (score < global_candidates[cnt]["score"]):
                    global_candidates[cnt] = {"label": label, "score": score, "Zin": top_z_cpu[i], "total": total}

            # グローバル辞書から改めて上位kを作る（安定化）
            items = list(global_candidates.items())
            items.sort(key=lambda kv: kv[1]["score"])
            top_items = items[: cfg.top_k]

            global_best_scores  = np.asarray([it[1]["score"] for it in top_items], dtype=float)
            global_best_records = [{"label": it[1]["label"], "score": it[1]["score"], "total_parts": it[1]["total"]} for it in top_items]
            global_best_traces  = [(it[1]["label"], it[1]["Zin"]) for it in top_items]

            # ライブ描画（プレースホルダ→実データに置き換え）
            if cfg.plot_view:
                eval_lines = {
                    "band": (cfg.f_L, cfg.f_H),
                    "mask_x": target_poly[:, 0] if target_poly is not None else None,
                    "mask_y": target_poly[:, 1] if target_poly is not None else None,
                }
                ts, ax2 = live_update_topk(
                    figs.get("ax2"),
                    f_cpu,
                    global_best_traces,
                    eval_lines,
                    now_ts=time.time(),
                    min_interval=cfg.plot_min_interval_s,
                    force_interval=cfg.plot_force_update_interval_s,
                )
                figs["ax2"] = ax2
                figs["fig2"] = None if ax2 is None else ax2.figure
                last_plot_ts = ts

        pending: List[np.ndarray] = []
        processed_total = 0

        try:
            for vec_cpu in enumerate_count_vectors(
                n_types=len(cfg.capacitors),
                max_total=cfg.max_total_parts,
                min_ratio=cfg.min_total_parts_ratio,
                shuffle=cfg.shuffle_evaluation,
                seed=cfg.seed,
            ):
                pending.append(vec_cpu)

                # Heartbeat：チャンク処理待ちでも plot_force_update_interval_s ごとに再描画
                if cfg.plot_view:
                    now = time.time()
                    ax2 = figs.get("ax2")
                    last = getattr(ax2, "_last_update_ts", 0.0) if ax2 is not None else 0.0
                    if (now - last) >= float(cfg.plot_force_update_interval_s):
                        if global_best_traces:
                            # 既存Top-kを再描画（min/force=0で強制更新）
                            _, ax2 = live_update_topk(ax2, f_cpu, global_best_traces, eval_lines,
                                                      now_ts=now, min_interval=0.0, force_interval=0.0)
                        else:
                            # まだ候補無しのときはプレースホルダを再描画
                            _, ax2 = prepare_fig2(ax2, f_cpu, eval_lines, "Searching in progress...")
                        figs["ax2"] = ax2
                        figs["fig2"] = None if ax2 is None else ax2.figure
                        last_plot_ts = now

                # しきい値でチャンク処理
                if len(pending) >= flush_threshold:
                    batch = np.asarray(pending, dtype=np.int16)
                    pending.clear()
                    counts_dev = batch_to_device(batch, xp)

                    # PDN 合成 & スコア
                    Zin_dev = zin_batch(f_dev, Zc_list_dev, counts_dev, parasitics, order_idx,
                                        cap_specs=cfg.capacitors, xp=xp, dtype_c=dtype_c)
                    mets = metrics_from_zin(Zin_dev, f_dev, m_eval, z_target, xp)
                    parts_count = counts_dev.sum(axis=1)
                    base_score = score_linear_comb(mets, cfg.weights, parts_count, xp)
                    if cfg.mc_enable and cfg.mc_samples > 0:
                        mc_worst = monte_carlo_worst(Zc_list_dev, counts_dev, cfg, f_dev,
                                                     parasitics, order_idx, cap_specs=cfg.capacitors, xp=xp, dtype_c=dtype_c)
                        base_score = base_score + cfg.weights.get("mc_worst", 1.0) * mc_worst

                    _merge_topk_and_plot(base_score, Zin_dev, counts_dev)

                    # ここでチャンク分だけ進捗を更新（要求どおり）
                    processed_total += int(batch.shape[0])
                    pbar.update(int(batch.shape[0]))
                    if (pbar.n % 1024) == 0:
                        pbar.refresh()

                    if use_gpu:
                        try:
                            import cupy as _cp
                            _cp.get_default_memory_pool().free_all_blocks()
                            _cp.get_default_pinned_memory_pool().free_all_blocks()
                        except Exception:
                            pass

            # 端数バッチ
            if pending:
                batch = np.asarray(pending, dtype=np.int16)
                pending.clear()
                counts_dev = batch_to_device(batch, xp)

                Zin_dev = zin_batch(f_dev, Zc_list_dev, counts_dev, parasitics, order_idx,
                                    cap_specs=cfg.capacitors, xp=xp, dtype_c=dtype_c)
                mets = metrics_from_zin(Zin_dev, f_dev, m_eval, z_target, xp)
                parts_count = counts_dev.sum(axis=1)
                base_score = score_linear_comb(mets, cfg.weights, parts_count, xp)
                if cfg.mc_enable and cfg.mc_samples > 0:
                    mc_worst = monte_carlo_worst(Zc_list_dev, counts_dev, cfg, f_dev,
                                                 parasitics, order_idx, cap_specs=cfg.capacitors, xp=xp, dtype_c=dtype_c)
                    base_score = base_score + cfg.weights.get("mc_worst", 1.0) * mc_worst

                _merge_topk_and_plot(base_score, Zin_dev, counts_dev)

                processed_total += int(batch.shape[0])
                pbar.update(int(batch.shape[0]))
                pbar.refresh()

                if use_gpu:
                    try:
                        import cupy as _cp
                        _cp.get_default_memory_pool().free_all_blocks()
                        _cp.get_default_pinned_memory_pool().free_all_blocks()
                    except Exception:
                        pass

        except KeyboardInterrupt:
            logger.warning("探索がユーザーにより中断されました。上位候補を出力します。")
        finally:
            pbar.refresh()
            pbar.close()

        if cfg.plot_view and global_best_traces:
            eval_lines = {
                "band": (cfg.f_L, cfg.f_H),
                "mask_x": target_poly[:, 0] if target_poly is not None else None,
                "mask_y": target_poly[:, 1] if target_poly is not None else None,
            }
            # min_interval/force_interval を 0 にしてレート制御を無効化
            import time as _time
            _, ax2 = live_update_topk(
                figs.get("ax2"),
                f_cpu,
                global_best_traces,
                eval_lines,
                now_ts=_time.time(),
                min_interval=0.0,
                force_interval=0.0,
            )
            figs["ax2"] = ax2
            figs["fig2"] = None if ax2 is None else ax2.figure

        # 11 Excel 出力
        if global_best_traces:
            topk_series = [(lab, decimate_for_transfer(tr, factor=1)) for lab, tr in global_best_traces]
            out_path = write_topk_result(cfg.excel_path, cfg.excel_name, f_cpu,
                                         topk_series, global_best_records, target_poly)
            logger.info(f"Excel 出力: {out_path}")
        else:
            logger.warning("上位候補がありませんでした。")

        # 12 ブロッキング表示
        if cfg.plot_view and figs.get("fig1") is not None:
            import matplotlib.pyplot as plt
            logger.info("グラフを閉じると終了します。")
            plt.ioff(); plt.show()

    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decoupling capacitor exhaustive search")
    parser.add_argument("--toml", type=str, default=None, help="Path to user_config.toml (optional)")
    args = parser.parse_args()
    main(args.toml)

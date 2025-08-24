"""
XlsxWriter で結果出力（Top-k |Zin| を 1 枚の散布図で両軸ログ）
- write_topk_result(excel_path, excel_name, f_cpu, topk_series, score_table, target_polyline) -> Path
"""
from __future__ import annotations

from typing import List, Tuple, Dict
from pathlib import Path
import datetime as dt
import numpy as np
import xlsxwriter  # XlsxWriter

def write_topk_result(
    excel_path: str | None,
    excel_name: str,
    f_cpu: np.ndarray,
    topk_series: List[Tuple[str, np.ndarray]],
    score_table: List[Dict],
    target_polyline: np.ndarray | None           # shape (K,2) [[f, Z], ...] or None
) -> Path:
    """
    戻り値: 保存先ファイルパス
    - X/Y とも log 軸。散布図 + lines（マーカー無し）
    - 目標マスク（点線）
    """
    out_dir = Path(excel_path) if excel_path else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d%H%M")
    out_file = out_dir / f"{excel_name}_{ts}.xlsx"

    wb = xlsxwriter.Workbook(out_file.as_posix(), {"nan_inf_to_errors": True})
    try:
        # --- シート1: データ ---
        ws = wb.add_worksheet("TopK_Data")
        ws.write(0, 0, "f [Hz]")
        for i, f in enumerate(f_cpu, start=1):
            ws.write_number(i, 0, float(f))

        for col, (label, Zin) in enumerate(topk_series, start=1):
            ws.write(0, col, label)
            for r, v in enumerate(np.abs(Zin), start=1):
                ws.write_number(r, col, float(v))

        mask_cols = None
        if target_polyline is not None and len(target_polyline) >= 2:
            mask_cols = (len(topk_series) + 2, len(topk_series) + 3)
            ws.write(0, mask_cols[0], "Mask f")
            ws.write(0, mask_cols[1], "Mask Z")
            for r, (ff, zz) in enumerate(target_polyline, start=1):
                ws.write_number(r, mask_cols[0], float(ff))
                ws.write_number(r, mask_cols[1], float(zz))

        # --- シート2: スコア ---
        ws2 = wb.add_worksheet("TopK_Score")
        if score_table:
            headers = list(score_table[0].keys())
            for c, h in enumerate(headers):
                ws2.write(0, c, h)
            for r, row in enumerate(score_table, start=1):
                for c, h in enumerate(headers):
                    val = row.get(h, "")
                    if isinstance(val, (int, float)):
                        ws2.write_number(r, c, float(val))
                    else:
                        ws2.write(r, c, str(val))

        # --- グラフ（Scatter のみ X-log が効く） ---
        chart = wb.add_chart({"type": "scatter"})
        chart.set_legend({"position": "top"})
        chart.set_title({"name": "Top-k |Zin| (Log-Log)"})
        chart.set_x_axis({"name": "Frequency [Hz]", "log_base": 10})
        chart.set_y_axis({"name": "|Zin| [Ohm]", "log_base": 10})

        n_rows = len(f_cpu)
        for idx, (label, _Zin) in enumerate(topk_series):
            series = {
                "name": [ws.name, 0, idx + 1],
                "categories": [ws.name, 1, 0, n_rows, 0],        # f
                "values":     [ws.name, 1, idx + 1, n_rows, idx + 1],  # |Z|
                "marker": {"type": "none"},
                "line": {"width": 1.25},
            }
            chart.add_series(series)

        if mask_cols and target_polyline is not None:
            mcx, mcy = mask_cols
            series_mask = {
                "name": "Target mask",
                "categories": [ws.name, 1, mcx, 1 + len(target_polyline) - 1, mcx],
                "values":     [ws.name, 1, mcy, 1 + len(target_polyline) - 1, mcy],
                "marker": {"type": "none"},
                "line": {"dash_type": "dash", "width": 1.0, "color": "black"},
            }
            chart.add_series(series_mask)

        ws.insert_chart(1, len(topk_series) + 3, chart, {"x_offset": 10, "y_offset": 10})
    finally:
        wb.close()

    return out_file

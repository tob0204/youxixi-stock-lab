"""油洗洗：逐筆交易日證據。日期、訊號與勝負都可由使用者追溯。"""
from __future__ import annotations

from html import escape
import math
import pandas as pd
import numpy as np

from window_engine import cutoff_for_window, WINDOW_LABELS


def build_ledger(prices: pd.DataFrame, signals: dict[str, pd.Series],
                 horizons: list[int] | tuple[int, ...], round_trip_cost_pct: float) -> pd.DataFrame:
    """一筆=已完成的日線事件。t 收盤觸發，t+1 開盤假設進場，t+h 收盤出場。

    這是對稱方向代理模型；看跌不等於真實合法放空交易。
    """
    n = len(prices)
    if n < 2:
        return pd.DataFrame()
    days = pd.DatetimeIndex(prices.index)
    opens = prices['Open'].to_numpy(float)
    closes = prices['Close'].to_numpy(float)
    output: list[dict] = []
    for name, series in signals.items():
        vec = series.reindex(days).fillna(0).to_numpy(dtype=int)
        for horizon in horizons:
            for t in np.flatnonzero(vec):
                exit_at = t + horizon
                if exit_at >= n or t + 1 >= n:
                    continue
                direction = int(vec[t])
                gross = direction * (closes[exit_at] / opens[t + 1] - 1) * 100
                net = gross - float(round_trip_cost_pct)
                output.append({
                    '技術指標': name,
                    '方向': '看漲' if direction > 0 else '看跌',
                    '預測交易日': horizon,
                    '訊號日': days[t].strftime('%Y-%m-%d'),
                    '進場日': days[t+1].strftime('%Y-%m-%d'),
                    '結果日': days[exit_at].strftime('%Y-%m-%d'),
                    '訊號收盤價': float(closes[t]),
                    '下一日開盤價': float(opens[t+1]),
                    '結果日收盤價': float(closes[exit_at]),
                    '隔日跳空%': 100 * (opens[t+1]/closes[t]-1),
                    '隔日收盤變動%': 100 * (closes[t+1]/closes[t]-1),
                    '方向報酬%': float(gross),
                    '扣成本後報酬%': float(net),
                    '是否成功': bool(net > 0),
                })
    if not output:
        return pd.DataFrame(columns=['技術指標','方向','預測交易日','訊號日','進場日','結果日','是否成功'])
    return pd.DataFrame(output).sort_values(['訊號日','技術指標','方向','預測交易日']).reset_index(drop=True)


def select_ledger(ledger: pd.DataFrame, indicator: str, direction: str,
                  horizon: int, asof: pd.Timestamp, label: str) -> pd.DataFrame:
    if ledger.empty:
        return ledger.copy()
    start = cutoff_for_window(pd.Timestamp(asof), label)
    dt = pd.to_datetime(ledger['訊號日'])
    outcome = pd.to_datetime(ledger['結果日'])
    return ledger.loc[(ledger['技術指標'] == indicator) &
                      (ledger['方向'] == direction) &
                      (ledger['預測交易日'] == horizon) &
                      (dt >= start) & (outcome <= pd.Timestamp(asof))].copy()


def dates_tooltip(ledger: pd.DataFrame, max_chars: int = 3500) -> str:
    """逐筆日期皆列出：如果長到瀏覽器無法閱讀，提示改在明細表看完整資料。"""
    if ledger.empty:
        return '無已完成樣本'
    lines = [f"{row['訊號日']} → {row['結果日']} {'成功' if row['是否成功'] else '失敗'} ({row['扣成本後報酬%']:+.2f}%)" for _, row in ledger.iterrows()]
    output = '\n'.join(lines)
    if len(output) > max_chars:
        shortened = output[:max_chars].rsplit('\n',1)[0]
        return shortened + f'\n...共 {len(lines)} 筆；請看下方完整樣本明細'
    return output


def summary_ledger_table(frame: pd.DataFrame, ledger: pd.DataFrame, asof: pd.Timestamp,
                         count_column='訊號數', height_max=700) -> str:
    """產生真正逐儲存格 hover，避開 st.dataframe 只有表頭 tooltip 的限制。"""
    if frame.empty:
        return '<p>沒有回測結果</p>'
    showing = [c for c in ['回測期間','技術指標','方向','預測交易日',count_column,
                           '成本後成功次數','成本後命中率%（簡化）','近期加權命中率%'] if c in frame.columns]
    parts = ['''<style>
    .yxs-table-wrap{overflow:auto;max-height:%spx;border:1px solid #d9dce1;border-radius:9px;margin:10px 0}
    .yxs-table{width:100%%;border-collapse:collapse;font-family:inherit;font-size:13px;}
    .yxs-table th{position:sticky;top:0;background:#f1f3f6;color:#262b39;padding:9px;text-align:left;white-space:nowrap;z-index:1;}
    .yxs-table td{padding:8px;border-bottom:1px solid #e3e5e8;white-space:nowrap;color:#232b37;}
    .yxs-table tr:hover{background:#eef5ff;}
    .yxs-count{cursor:help;text-decoration:underline dotted #2563eb;text-underline-offset:4px;color:#1761be;font-weight:750;}
    </style><div class="yxs-table-wrap"><table class="yxs-table"><thead><tr>''' % int(height_max)]
    for col in showing:
        parts.append(f'<th>{escape(col)}{(" ⓘ" if col == count_column else "")}</th>')
    parts.append('</tr></thead><tbody>')
    for _, row in frame.iterrows():
        parts.append('<tr>')
        for col in showing:
            val = row[col]
            if col == count_column:
                chosen = select_ledger(ledger, str(row['技術指標']), str(row['方向']),
                                       int(row['預測交易日']), asof, str(row['回測期間']))
                tooltip = escape(dates_tooltip(chosen), quote=True)
                parts.append(f'<td><span class="yxs-count" title="{tooltip}">{int(val)}</span></td>')
            elif isinstance(val, (float,np.floating)):
                parts.append(f'<td>{val:,.1f}' + ('%' if '%' in col else '') + '</td>' if math.isfinite(val) else '<td>—</td>')
            else:
                parts.append(f'<td>{escape(str(val))}</td>')
        parts.append('</tr>')
    parts.append('</tbody></table></div><small>ⓘ 滑鼠停在帶底線的樣本數上，可看到訊號日、結果日與每次成功／失敗。完整明細可在下方篩選下載。</small>')
    return ''.join(parts)

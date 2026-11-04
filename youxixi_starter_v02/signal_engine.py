"""油洗洗 v0.6：可獨立測試的日線訊號回測與最新決策引擎。

注意：這裡的歷史成功率是已完成交易日訊號的**樣本內**頻率，
不是當下經校準的條件機率；不提供即時當沖交易與自動下單。
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd


def wilson_95(wins: int, count: int) -> tuple[float, float]:
    """二元觀測值的 Wilson 95% 近似區間；不處理事件重疊/市場非定態。"""
    if count <= 0:
        return (math.nan, math.nan)
    z = 1.96
    p = wins / count
    denom = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denom
    width = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count ** 2)) / denom
    return max(0.0, center - width), min(1.0, center + width)


def evaluate(prices: pd.DataFrame, signals: Mapping[str, pd.Series],
             cutoff: pd.Timestamp, horizons: list[int], half_life: int,
             round_trip_cost_pct: float) -> pd.DataFrame:
    """訊號 t 收盤後可知，t+1 開盤進場，t+h 收盤出場；所有已知標籤才納入。

    看漲/看跌方向報酬只是對稱的代理計算，並非真實做多/做空報酬。
    交易日以 DataFrame 列序代表（正確處理假日）。
    """
    if half_life < 1:
        raise ValueError("近期加權半衰期必須 >= 1")
    if not 0 <= round_trip_cost_pct <= 100:
        raise ValueError("往返成本必須介於 0 到 100%")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("行情需按日期由舊到新排序")
    opens = prices["Open"].to_numpy(dtype=float)
    closes = prices["Close"].to_numpy(dtype=float)
    day_index = np.arange(len(prices), dtype=int)
    in_backtest = np.asarray(prices.index >= cutoff)
    rows: list[dict] = []
    for horizon in horizons:
        if horizon < 1:
            raise ValueError("預測期限至少 1 個交易日")
        eligible = day_index[in_backtest & (day_index + horizon < len(prices))]
        raw_ret = closes[eligible + horizon] / opens[eligible + 1] - 1 if len(eligible) else np.array([])
        for indicator, series in signals.items():
            signal_values = series.reindex(prices.index).fillna(0).to_numpy(dtype=int)
            for direction, direction_name in ((1, "看漲"), (-1, "看跌")):
                positions = np.flatnonzero(signal_values[eligible] == direction)
                count = int(len(positions))
                result = {
                    "技術指標": indicator, "方向": direction_name, "預測交易日": horizon,
                    "訊號數": count, "方向命中次數": 0, "成本後成功次數": 0,
                    "歷史方向命中率%": math.nan, "成本後命中率%（簡化）": math.nan,
                    "近期加權命中率%": math.nan, "近期有效樣本數": math.nan,
                    "95% Wilson 區間（成本後）": "—",
                    "平均方向報酬%（未扣成本）": math.nan,
                    "平均方向報酬%（簡化扣成本）": math.nan,
                }
                if count > 0:
                    gross = raw_ret[positions] * direction
                    net = gross - round_trip_cost_pct / 100.0
                    gross_wins = int(np.count_nonzero(gross > 0))
                    net_flags = (net > 0).astype(float)
                    net_wins = int(net_flags.sum())
                    low, high = wilson_95(net_wins, count)
                    # 標籤完成 t+h 的日序，避免把未完成的結果拿來加權。
                    label_ages = (len(prices) - 1) - (eligible[positions] + horizon)
                    weights = 0.5 ** (label_ages / half_life)
                    weighted = float(np.dot(weights, net_flags) / weights.sum())
                    eff_n = float(weights.sum() ** 2 / np.dot(weights, weights))
                    result.update({
                        "方向命中次數": gross_wins, "成本後成功次數": net_wins,
                        "歷史方向命中率%": float(100 * gross_wins / count),
                        "成本後命中率%（簡化）": float(100 * net_wins / count),
                        "近期加權命中率%": float(100 * weighted),
                        "近期有效樣本數": eff_n,
                        "95% Wilson 區間（成本後）": f"{100 * low:.0f}%–{100 * high:.0f}%",
                        "平均方向報酬%（未扣成本）": float(gross.mean() * 100),
                        "平均方向報酬%（簡化扣成本）": float(net.mean() * 100),
                    })
                rows.append(result)
    return pd.DataFrame(rows)


def decide_latest(signals: Mapping[str, pd.Series], results: pd.DataFrame,
                  horizon: int, threshold_pct: float = 90.0,
                  min_samples: int = 20, min_effective_samples: float = 5.0,
                  require_recent: bool = True,
                  allow_signal: bool = True) -> tuple[str, pd.DataFrame, str]:
    """判斷當日收盤後單一新事件：買、賣同時成立則觀望。

    決策條件：當日有新事件；樣本數達最低；成本後歷史命中率 *嚴格大於* 門檻；
    若 require_recent，再要求近期加權率 *嚴格大於* 門檻及有效樣本數達門檻。
    即使達標也是基於歷史頻率的研究標籤，不是可保證的機率。
    """
    if not 0 <= threshold_pct <= 100:
        raise ValueError("門檻必須介於 0～100")
    if min_samples < 1 or min_effective_samples < 1:
        raise ValueError("樣本數門檻需大於等於 1")
    if not signals:
        return "觀望", pd.DataFrame(), "沒有選取技術指標。"
    records = []
    approved_buy, approved_sell = [], []
    for name, series in signals.items():
        current = int(series.iloc[-1]) if len(series) else 0
        direction = "看漲" if current == 1 else "看跌" if current == -1 else "無新訊號"
        subset = results.loc[(results["技術指標"] == name) &
                             (results["預測交易日"] == horizon) &
                             (results["方向"] == direction)]
        stats = subset.iloc[0] if len(subset) else None
        n = int(stats["訊號數"]) if stats is not None else 0
        rate = float(stats["成本後命中率%（簡化）"]) if n else math.nan
        weighted = float(stats["近期加權命中率%"]) if n else math.nan
        effective = float(stats["近期有效樣本數"]) if n else math.nan
        eligible = current != 0 and n >= min_samples and rate > threshold_pct
        if require_recent:
            eligible = eligible and effective >= min_effective_samples and weighted > threshold_pct
        if not allow_signal:
            eligible = False
        if current == 0:
            reason = "最新日線沒有新交叉／突破"
        elif not allow_signal:
            reason = "非最新有效日線；禁止視為即時訊號"
        elif n < min_samples:
            reason = f"歷史樣本不足（{n} / {min_samples}）"
        elif not rate > threshold_pct:
            reason = f"成本後命中率未高於 {threshold_pct:g}%"
        elif require_recent and effective < min_effective_samples:
            reason = f"近期有效樣本不足（{effective:.1f} / {min_effective_samples:g}）"
        elif require_recent and not weighted > threshold_pct:
            reason = f"近期加權命中率未高於 {threshold_pct:g}%"
        else:
            reason = "符合條件"
        records.append({
            "技術指標": name, "最新訊號": direction, "歷史樣本數": n,
            "成本後歷史成功率%": round(rate, 2) if np.isfinite(rate) else None,
            "近期加權成功率%": round(weighted, 2) if np.isfinite(weighted) else None,
            "近期有效樣本數": round(effective, 1) if np.isfinite(effective) else None,
            "符合買賣門檻": "是" if eligible else "否", "說明": reason,
        })
        if eligible:
            (approved_buy if current == 1 else approved_sell).append(name)
    table = pd.DataFrame(records)
    if approved_buy and approved_sell:
        return "觀望", table, "多空訊號同時超過門檻，為避免互相矛盾，採觀望。"
    if approved_buy:
        return "買入", table, "符合條件的看漲指標：" + "、".join(approved_buy)
    if approved_sell:
        return "賣出", table, "符合條件的看跌指標：" + "、".join(approved_sell)
    return "觀望", table, "當前沒有任何技術指標同時通過樣本數與成功率篩選。"

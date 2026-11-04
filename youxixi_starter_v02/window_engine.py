"""油洗洗 v0.7：多期間回測比較與次日方向研究判斷（非實盤勝率校準）。"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from signal_engine import evaluate, wilson_95

# 用實際交易日資料篩選，2 週採 14 個曆日（約 10 個交易日），不是固定 10 筆。
WINDOW_LABELS = ('2週', '1個月', '3個月', '6個月', '12個月')

def cutoff_for_window(last_day: pd.Timestamp, label: str) -> pd.Timestamp:
    end = pd.Timestamp(last_day)
    if label == '2週':
        return end - pd.Timedelta(days=14)
    month_lookup = {'1個月':1, '3個月':3, '6個月':6, '12個月':12}
    if label not in month_lookup:
        raise ValueError(f'未知回測期間：{label}')
    return end - pd.DateOffset(months=month_lookup[label])


def compare_windows(prices: pd.DataFrame, signals: dict[str, pd.Series],
                    horizons: list[int] | tuple[int,...], half_life: int,
                    round_trip_cost_pct: float) -> pd.DataFrame:
    """同一份歷史資料、同一套訊號與成本，比較五個固定回測窗；只算已完成的交易結果。"""
    if prices.empty:
        raise ValueError('沒有任何可用行情')
    frames=[]
    ref=prices.index.max()
    for label in WINDOW_LABELS:
        start = cutoff_for_window(ref,label)
        result = evaluate(prices, signals, start, list(horizons), half_life, round_trip_cost_pct)
        result.insert(0,'回測期間', label)
        result.insert(1,'回測起始日',start.date().isoformat())
        result.insert(2,'最後行情日',ref.date().isoformat())
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def recommend_across_windows(signals: dict[str, pd.Series], comparisons: pd.DataFrame,
                             horizon: int, threshold_pct: float=90.0,
                             min_samples: int=20, min_effective_samples: float=5.0,
                             require_recent: bool=True, allow_signal: bool=True
                             ) -> tuple[str,pd.DataFrame,str,dict | None]:
    """嚴格門檻：有當日訊號、樣本數、成本後勝率、正平均淨報酬及近期確認。

    期間篩選仍屬**樣本內**模型比較，不能當成獨立樣本外勝率預測。
    五個期間高度重疊，不得視為五份獨立驗證資料。
    """
    if not 0 <= threshold_pct <= 100:
        raise ValueError('勝率門檻須在 0～100%')
    if min_samples < 1 or min_effective_samples < 1:
        raise ValueError('最低樣本數必須至少 1')
    if not signals or comparisons.empty:
        return '觀望',pd.DataFrame(),'沒有足夠的訊號資料。',None
    rows=[]
    candidates=[]
    active_directions={}
    restricted = comparisons.loc[comparisons['預測交易日'].eq(horizon)]
    for indicator, series in signals.items():
        current = int(series.iloc[-1]) if len(series) else 0
        direction = '看漲' if current==1 else '看跌' if current==-1 else '無新訊號'
        active_directions[indicator]=current
        recent_warning=False
        recent = restricted.loc[(restricted['技術指標']==indicator) &
                                (restricted['方向']==direction) &
                                (restricted['回測期間']=='2週')]
        if len(recent):
            r = recent.iloc[0]
            rn=int(r['訊號數'])
            rr=float(r['成本後命中率%（簡化）']) if rn else math.nan
            # 近期小樣本發現顯著劣化時保守擋下；少於五筆只提示不足，不認定是反證。
            recent_warning=(rn>=5 and np.isfinite(rr) and rr < max(50.0,threshold_pct-30.0))
        for label in WINDOW_LABELS:
            sample=restricted.loc[(restricted['技術指標']==indicator) &
                                  (restricted['方向']==direction) &
                                  (restricted['回測期間']==label)]
            if sample.empty:
                # 最新日沒有新訊號者仍顯示 5 個期間，避免讓使用者以為漏算。
                rows.append({'技術指標':indicator,'最新訊號':direction,'回測期間':label,
                             '已完成樣本數':0,'成功次數':0,'成本後勝率%':None,
                             '近期加權勝率%':None,'近期有效樣本數':None,
                             '平均成本後報酬%':None,'Wilson 95% 下界%':None,
                             '評估':'今天沒有新訊號' if current==0 else '無相關歷史結果'})
                continue
            stat=sample.iloc[0]
            n=int(stat['訊號數'])
            win=int(stat['成本後成功次數'])
            rate=float(stat['成本後命中率%（簡化）']) if n else math.nan
            weighted=float(stat['近期加權命中率%']) if n else math.nan
            effective=float(stat['近期有效樣本數']) if n else math.nan
            avg=float(stat['平均方向報酬%（簡化扣成本）']) if n else math.nan
            lower=100*wilson_95(win,n)[0] if n else math.nan
            if current==0:
                reason='今天沒有新訊號'
            elif not allow_signal:
                reason='非最新有效收盤資料'
            elif n < min_samples:
                reason=f'樣本不足（{n}/{min_samples}）'
            elif not rate > threshold_pct:
                reason='勝率未高於門檻'
            elif not avg > 0:
                reason='平均成本後報酬未達正值'
            elif require_recent and effective < min_effective_samples:
                reason='近期加權有效樣本不足'
            elif require_recent and not weighted > threshold_pct:
                reason='近期加權勝率未達門檻'
            elif recent_warning:
                reason='近兩週較長期間顯示惡化警訊'
            else:
                reason='候選達標（樣本內）'
                candidates.append({'指標':indicator,'方向':direction,'回測期間':label,
                                   '勝率':rate,'樣本數':n,'近期勝率':weighted,
                                   '淨報酬':avg,'Wilson下界':lower,'成功次數':win})
            rows.append({'技術指標':indicator,'最新訊號':direction,'回測期間':label,
                         '已完成樣本數':n,'成功次數':win,
                         '成本後勝率%':round(rate,2) if np.isfinite(rate) else None,
                         '近期加權勝率%':round(weighted,2) if np.isfinite(weighted) else None,
                         '近期有效樣本數':round(effective,1) if np.isfinite(effective) else None,
                         '平均成本後報酬%':round(avg,3) if np.isfinite(avg) else None,
                         'Wilson 95% 下界%':round(lower,2) if np.isfinite(lower) else None,
                         '評估':reason})
    table=pd.DataFrame(rows)
    if not allow_signal:
        return '觀望',table,'目前並非可用的最新完整收盤日線，不能產生明日交易訊號。',None
    sides={c['方向'] for c in candidates}
    if len(sides)>1:
        return '觀望',table,'不同指標的合格買賣方向相互衝突，因此採觀望。',None
    if not candidates:
        return '觀望',table,'沒有任何最新訊號同時通過勝率、樣本數、平均淨報酬與近期確認；不能只挑最高勝率就下結論。',None
    # 不以單一最高勝率排名，先看 Wilson 信賴區間下界、再看樣本數。
    best=max(candidates,key=lambda c:(c['Wilson下界'],c['樣本數'],c['淨報酬']))
    side='買入' if best['方向']=='看漲' else '賣出'
    rows_best=table[(table['技術指標']==best['指標']) & (table['最新訊號']==best['方向'])]
    short=rows_best.loc[rows_best['回測期間']=='2週']
    if not short.empty:
        short=short.iloc[0]
        if int(short['已完成樣本數'])<min_samples:
            short_desc=f"兩週內僅 {int(short['已完成樣本數'])} 筆已完成同方向訊號，不足以確認或否定這個結論。"
        else:
            short_desc=f"兩週勝率 {short['成本後勝率%']}%，狀態：{short['評估']}。"
    else:
        short_desc='兩週期間沒有可比較資料。'
    why=(f"{best['指標']} 的最新日線出現{best['方向']}訊號；"
         f"主要依據 {best['回測期間']} 回測：{best['成功次數']}/{best['樣本數']} 次成功，"
         f"成本後歷史勝率 {best['勝率']:.1f}%（Wilson 95% 下界 {best['Wilson下界']:.1f}%），"
         f"平均簡化成本後報酬 {best['淨報酬']:.2f}%。{short_desc} "
         "期間互相重疊，這是探索性歷史比較，尚非樣本外驗證的預測機率。")
    return side,table,why,best

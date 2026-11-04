"""逐日前推（walk-forward）：在歷史每一日，只使用該日收盤前已完成的交易結果。

一日策略：t 日收盤才算訊號，t+1 開盤以預設限價假設成交，當日收盤平倉。
價格模型：以訓練事件之隔日開盤跳空分位數定限價；只有開盤穿越門檻才當作成交。
研究結果 ≠ 可即時成交，亦 ≠ 看空就可放空。
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from audit_engine import select_ledger
from signal_engine import wilson_95
from window_engine import WINDOW_LABELS


def tick_size(p: float) -> float:
    if p < 10: return .01
    if p < 50: return .05
    if p < 100: return .1
    if p < 500: return .5
    if p < 1000: return 1.
    return 5.


def tick_round(x: float, side: int, is_etf=False) -> float:
    """普通股參考台股跳動單位；此模型不適用 ETF／特殊商品。"""
    t=tick_size(x)
    return round((math.floor(x/t+1e-9) if side==1 else math.ceil(x/t-1e-9))*t,2)


def _hist_stats(data: pd.DataFrame, close: float, direction: int,
                half_life: int, index_map: dict[pd.Timestamp,int],
                today_index: int, price_q_buy: float=.60) -> dict:
    """先由歷史跳空分布提出限價，再只在當日開盤可滿足限價的案例評估勝率。"""
    n=len(data)
    if not n:
        return dict(total=0, filled=0, rate=float('nan'), rate_all=float('nan'),
                    weighted=float('nan'), eff_n=0., win=0, limit=None,
                    projected_close=None, low=None, avg=float('nan'), filled_ledger=pd.DataFrame(),
                    close_band=(None,None))
    gaps=data['隔日跳空%'].to_numpy(float)/100
    # buy limit higher allows more fills; sell threshold lower allows more fills
    quantile = price_q_buy if direction==1 else (1-price_q_buy)
    limit=tick_round(close*(1+float(np.quantile(gaps,quantile))),direction)
    if direction==1:
        filled_mask=(close*(1+gaps) <= limit +1e-8)
    else:
        filled_mask=(close*(1+gaps) >= limit -1e-8)
    filled=data.iloc[np.flatnonzero(filled_mask)]
    nfill=len(filled)
    nwin=int(filled['是否成功'].sum()) if nfill else 0
    ratio=nwin/nfill if nfill else float('nan')
    rr = filled['扣成本後報酬%'].to_numpy(float) if nfill else np.array([],float)
    avg=float(rr.mean()) if nfill else float('nan')
    if nfill:
        ages=np.array([today_index-index_map.get(pd.Timestamp(x),today_index)
                       for x in filled['結果日']],dtype=float)
        weights=2.0**(-np.maximum(ages,0)/half_life)
        weighted=float(np.dot(weights,filled['是否成功'].to_numpy(float))/weights.sum())
        eff=float(weights.sum()**2/np.dot(weights,weights))
        low,_=wilson_95(nwin,nfill)
    else:
        weighted=eff=0.
        low=0.
    # 下一交易日收盤估計採 close(t+1)/close(t) 的歷史條件中位數，非精準報價。
    returns=data['隔日收盤變動%'].to_numpy(float)/100
    projected=round(close*(1+float(np.median(returns))),2)
    q10,q90=np.quantile(returns,[0.1,0.9])
    return dict(total=n, filled=nfill, rate=100*ratio, rate_all=100*float(data['是否成功'].mean()),
                weighted=100*weighted,eff_n=eff,win=nwin,limit=limit,
                projected_close=projected,low=100*low,avg=avg,
                close_band=(round(close*(1+q10),2),round(close*(1+q90),2)),
                filled_ledger=filled.copy())


def predict_day(prices: pd.DataFrame, signals: dict[str,pd.Series], ledger: pd.DataFrame,
                decision_index: int, windows: tuple[str,...]|list[str],
                threshold: float, min_samples: int, half_life: int,
                min_effective_samples: float, require_recent: bool=True,
                require_current: bool=False) -> dict:
    """decision_index = T 交易日位置；交易／預測的是 T+1，T+1 完全不可見。

    `require_current` 不允許當次資料不是最新日期時展示即時建議（由前端控制）。
    """
    decision_index=int(decision_index)
    T=pd.Timestamp(prices.index[decision_index]); close=float(prices['Close'].iloc[decision_index])
    index_map={pd.Timestamp(d):i for i,d in enumerate(prices.index)}
    candidate_rows=[]
    active=[]
    for name, series in signals.items():
        s=int(series.iloc[decision_index])
        if s==0: continue
        active.append(name)
        direction='看漲' if s==1 else '看跌'
        for label in windows:
            # 此函式只處理 1 個交易日：過往事件結果 <= T，排除任何 D 或之後的資料。
            hist=select_ledger(ledger,name,direction,1,T,label)
            stats=_hist_stats(hist,close,s,half_life,index_map,decision_index)
            qualifies=(stats['filled']>=min_samples and stats['rate']>threshold
                       and stats['avg']>0 and (not require_recent or (
                           stats['weighted']>threshold and stats['eff_n']>=min_effective_samples)))
            candidate_rows.append({
                '技術指標':name,'方向':direction,'回測期間':label,
                '歷史同訊號數':stats['total'],'符合限價樣本數':stats['filled'],
                '成功次數':stats['win'],'限價成交後勝率%':stats['rate'],
                '整體訊號勝率%':stats['rate_all'],
                '近期加權勝率%':stats['weighted'],
                '近期有效樣本數':stats['eff_n'],
                '平均扣成本報酬%':stats['avg'],
                'Wilson 95% 下界%':stats['low'],
                '歷史估計限價':stats['limit'],
                '成交樣本明細':stats.get('filled_ledger',pd.DataFrame()),
                '預測次日收盤價':stats['projected_close'],
                '預測收盤區間10-90%':stats['close_band'],
                '達標':bool(qualifies),
                '樣本明細':hist,
            })
    good=[c for c in candidate_rows if c['達標']]
    sides={c['方向'] for c in good}
    best=(max(good,key=lambda x:(x['Wilson 95% 下界%'],x['符合限價樣本數'],x['平均扣成本報酬%'])) if good
          else None)
    if not active:
        verdict, reason='觀望','收盤時十種技術指標都沒有觸發新進場訊號。'
    elif not good:
        verdict, reason='觀望','目前各指標和回測期間都未通過最低有效樣本、勝率與近期表現檢查。'
    else:
        verdict='買入' if best['方向']=='看漲' else '賣出'
        reason=(f"綜合比較十種指標和五個期間，以 {best['技術指標']}／{best['回測期間']} 作為目前研究性候選；"
                f"{best['成功次數']}/{best['符合限價樣本數']} 筆成交假設成功。"
                + ('其他指標亦有相反方向的合格訊號，存在衝突，請審慎看待。' if len(sides)>1 else ''))

    # 也提供離門檻最近的「研究性候選」供觀望時查看，不能當成下單建議。
    preview=(max(candidate_rows,key=lambda x:(x['符合限價樣本數'],x['Wilson 95% 下界%'])) if candidate_rows else None)
    return {'日期':T.strftime('%Y-%m-%d'),'參考收盤價':close,'判斷':verdict,
            '原因':reason,'主要模型':best, '研究性候選':preview,
            '候選明細':candidate_rows}


def walk_forward(prices:pd.DataFrame, signals:dict[str,pd.Series], ledger:pd.DataFrame,
                 test_start:pd.Timestamp, windows:tuple[str,...]|list[str],
                 threshold:float, min_samples:int, half_life:int,
                 min_effective_samples:float, require_recent:bool=True,
                 round_trip_cost_pct:float=0.45) -> pd.DataFrame:
    """評分目標日 D：只能用 D 前一交易日 T 以前已完成的事件；D 的真實資料只用於驗證。"""
    results=[]
    days=prices.index
    for idx in range(1,len(days)):
        d=pd.Timestamp(days[idx])
        if d < pd.Timestamp(test_start): continue
        p=predict_day(prices,signals,ledger,idx-1,windows,threshold,min_samples,half_life,
                      min_effective_samples,require_recent)
        best=p['主要模型']
        signal_price=float(prices['Close'].iloc[idx-1])
        entry=float(prices['Open'].iloc[idx])
        end=float(prices['Close'].iloc[idx])
        status=p['判斷']
        if status=='買入' and best:
            fills= entry <= best['歷史估計限價'] +1e-8
            real_pnl=100*(end/entry-1)
        elif status=='賣出' and best:
            fills= entry >= best['歷史估計限價'] -1e-8
            real_pnl=100*(1-end/entry)
        else:
            fills=False
            real_pnl=float('nan')
        if fills:
            real_pnl -= float(round_trip_cost_pct)
        else:
            real_pnl = float('nan')
        results.append({'預測目標日':d.strftime('%Y-%m-%d'),
                        '訓練截止日':p['日期'],'訓練窗':'、'.join(windows),
                        '指標':best['技術指標'] if best else '',
                        '採用回測期間':best['回測期間'] if best else '',
                        '判斷':status,'昨日收盤價':signal_price,
                        '預估今日收盤價':best['預測次日收盤價'] if best else np.nan,
                        '實際今日開盤價':entry,'實際今日收盤價':end,
                        '建議限價':best['歷史估計限價'] if best else np.nan,
                        '限價符合開盤條件':bool(fills),
                        '依開盤成交假設扣成本報酬%':real_pnl,
                        '實際是否成功':(bool(real_pnl>0) if fills else None),
                        '預估收盤價誤差%':(100*(best['預測次日收盤價']/end-1) if best else np.nan),
                        '限價樣本':best['符合限價樣本數'] if best else 0,
                        '歷史限價勝率%':best['限價成交後勝率%'] if best else np.nan})
    return pd.DataFrame(results)

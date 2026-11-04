"""分批探索全台股；不把未掃描或抓取失敗的股票當作『觀望』。"""
from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from audit_engine import build_ledger
from forward_engine import predict_day
from indicators import compute_indicators, make_signals


def valid_universe(names:dict) -> list[str]:
    """限上市／上櫃普通股票的四位數代碼，排除 ETF 與未確認市場。"""
    return sorted([code for code,(_,market) in names.items() if len(code)==4 and code.isdigit()
                   and code[0]!='0' and market in ('上市','上櫃')])


def unpack_prices(downloaded:pd.DataFrame, symbol:str, total:int) -> pd.DataFrame:
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()
    if isinstance(downloaded.columns,pd.MultiIndex):
        if symbol in downloaded.columns.get_level_values(0):
            frame=downloaded[symbol].copy()
        elif symbol in downloaded.columns.get_level_values(1):
            frame=downloaded.xs(symbol,axis=1,level=1).copy()
        else:
            return pd.DataFrame()
    elif total==1:
        frame=downloaded.copy()
    else:
        return pd.DataFrame()
    if not set(('Open','High','Low','Close','Volume')).issubset(frame.columns):
        return pd.DataFrame()
    frame=frame[['Open','High','Low','Close','Volume']].copy()
    idx=pd.DatetimeIndex(pd.to_datetime(frame.index))
    if idx.tz is not None: idx=idx.tz_convert('Asia/Taipei').tz_localize(None)
    frame.index=idx.normalize()
    frame=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=['Open','High','Low','Close'])
    frame=frame[(frame['Open']>0)&(frame['Close']>0)]
    return frame.loc[~frame.index.duplicated(keep='last')].sort_index()


def scan_batch(codes:list[str], names:dict, fetch_start:str, end_exclusive:str,
               asof:pd.Timestamp, params:dict, chosen:list[str],
               threshold:float, min_n:int, half_life:int,eff_n:float,
               require_recent:bool, cost_pct:float) -> list[dict]:
    """只回傳已掃描股票與失敗說明；所有成功檢查均需訓練資料已完成。"""
    syms=[f'{c}.TWO' if names[c][1]=='上櫃' else f'{c}.TW' for c in codes]
    try:
        downloaded=yf.download(tickers=syms,start=fetch_start,end=end_exclusive,
                               interval='1d',auto_adjust=True,group_by='ticker',
                               progress=False,threads=True,timeout=25)
    except Exception as exc:
        return [{'股票代號':c,'股票名稱':names[c][0],'狀態':'讀取失敗','原因':str(exc)[:120]}
                for c in codes]
    rows=[]
    from datetime import datetime, time
    from zoneinfo import ZoneInfo
    now=datetime.now(ZoneInfo('Asia/Taipei'))
    for code,sym in zip(codes,syms):
        name=names[code][0]
        try:
            p=unpack_prices(downloaded,sym,len(syms))
            p=p.loc[p.index <= pd.Timestamp(asof)]
            if not p.empty and now.weekday()<5 and now.time()>=time(14,30) and p.index[-1].date()!=now.date():
                raise ValueError('今日收盤行情尚未更新')
            if len(p)<160:
                raise ValueError(f'歷史日K僅 {len(p)} 根，低於指標暖機所需的 160 根；不是股票不存在')
            values=compute_indicators(p,params)
            sig=make_signals(values,params)  # 完整策略規則；舊版只比對十指標名稱會篩成空字典
            ledger=build_ledger(p,sig,[1],cost_pct)
            result=predict_day(p,sig,ledger,len(p)-1,('2週','1個月','3個月','6個月','12個月'),
                               threshold,min_n,half_life,eff_n,require_recent)
            best=result['主要模型']
            rows.append({'股票代號':code,'股票名稱':name,'市場':names[code][1],
                         '訊號日':result['日期'],'收盤價':result['參考收盤價'],
                         '建議':result['判斷'],'指標':best['技術指標'] if best else '',
                         '回測期間':best['回測期間'] if best else '',
                         '建議限價':best['歷史估計限價'] if best else np.nan,
                         '價格條件勝率%':best['限價成交後勝率%'] if best else np.nan,
                         '有效樣本數':best['符合限價樣本數'] if best else 0,
                         '狀態':'成功','原因':result['原因']})
        except Exception as exc:
            rows.append({'股票代號':code,'股票名稱':name,'狀態':'讀取失敗','原因':str(exc)[:130]})
    return rows

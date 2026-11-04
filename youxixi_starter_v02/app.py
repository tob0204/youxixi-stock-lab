"""油洗洗 v0.8 | GitHub + Streamlit Cloud 零本機安裝。

收盤後 T -> 下一個交易日 T+1，開盤假設限價成交 -> 當日收盤平倉。
逐日前推驗證：對每個已知 D，模型只可使用 D 前一個交易日 T 及以前
**已完成**的訊號與結果。歷史成功率不是經校準的未來機率。
"""
from __future__ import annotations

from datetime import datetime, timedelta, time
from html import escape
from zoneinfo import ZoneInfo
import inspect
import math
import re

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from indicators import DEFAULTS, NAMES, compute_indicators, make_signals
from formulas import formula_sections
from stock_names import fetch_stock_names, find_stock_name
from window_engine import WINDOW_LABELS
from audit_engine import build_ledger, dates_tooltip
from forward_engine import predict_day, walk_forward
from scan_engine import valid_universe, scan_batch

VERSION = "v0.8"
COST_PCT = 0.45  # 示範用，非券商實際費率；均為報酬率百分點。
RECENT_HALFLIFE = 20  # 開發者自動風控，不要求使用者調整。
RECENT_MIN_EFFECTIVE_N = 5
HORIZON = 1
st.set_page_config(page_title="油洗洗｜隔日訊號回測", page_icon="📈", layout="wide")

@st.cache_data(ttl=3600, show_spinner=False)
def stock_directory():
    return fetch_stock_names()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_daily(code: str, market: str, start: str, end_exclusive: str):
    if "." in code:
        symbols=[code]
    else:
        symbols=([code+".TWO",code+".TW"] if market=="上櫃" else
                 [code+".TW",code+".TWO"])
    errs=[]
    for ticker in symbols:
        try:
            p=yf.Ticker(ticker).history(start=start,end=end_exclusive,interval="1d",
                                         auto_adjust=True,actions=False,timeout=20)
            if p is None or p.empty or not {"Open","High","Low","Close","Volume"}.issubset(p.columns):
                errs.append(ticker+": 缺資料")
                continue
            p=p[["Open","High","Low","Close","Volume"]].copy()
            ind=pd.DatetimeIndex(pd.to_datetime(p.index))
            if ind.tz is not None:
                ind=ind.tz_convert("Asia/Taipei").tz_localize(None)
            p.index=ind.normalize()
            p=p.loc[~p.index.duplicated(keep="last")].sort_index()
            p=p.replace([np.inf,-np.inf],np.nan).dropna(subset=["Open","High","Low","Close"])
            p=p.loc[(p["Open"]>0)&(p["Close"]>0)]
            if len(p)<380:
                errs.append(ticker+": 歷史日線不足 380 根")
                continue
            return p,ticker
        except Exception as e:
            errs.append(ticker+": "+str(e)[:75])
    raise ValueError("無法取得充分行情。"+"；".join(errs))


def format_money(value, suffix="元"):
    try:
        if value is None or not np.isfinite(float(value)): return "—"
        return f"{float(value):,.2f} {suffix}"
    except (TypeError, ValueError):
        return "—"


def candidates_frame(candidates:list[dict]):
    rows=[]
    for m in candidates:
        rows.append({
            "技術指標":m["技術指標"],"方向":m["方向"],"回測期間":m["回測期間"],
            "歷史同訊號":m["歷史同訊號數"],"符合限價樣本":m["符合限價樣本數"],
            "成功筆數":m["成功次數"],"樣本內勝率%":m["限價成交後勝率%"],
            "近期驗證%":m["近期加權勝率%"],
            "估計限價":m["歷史估計限價"],"平均淨報酬%":m["平均扣成本報酬%"],
            "判讀":"可供研究" if m["達標"] else (
                "樣本不足" if m["符合限價樣本數"] < int(st.session_state.get('min_samples',20)) else
                "未通過勝率或近期驗證"),
        })
    return pd.DataFrame(rows)


def hover_candidates_html(candidates:list[dict], cutoff=90):
    """原生 HTML title 支援每一格不同日期；另有完整版可篩選。"""
    if not candidates:
        return "<p>最新收盤尚未出現任何技術指標事件。</p>"
    rows=["""<style>
    .yy-scroll{overflow:auto;max-height:600px;border:1px solid #d2d7df;border-radius:9px;}
    table.yy{border-collapse:collapse;width:100%;font-size:13px;}
    .yy th{position:sticky;top:0;z-index:1;background:#eef2f6;color:#172332;text-align:left;padding:10px;white-space:nowrap;}
    .yy td{border-bottom:1px solid #e3e8ed;padding:9px;color:#253041;white-space:nowrap;}
    .yy tr:hover{background:#edf6ff;}
    .yy-tip{border-bottom:1px dotted #176fc0;color:#1259a0;cursor:help;font-weight:700;}
    </style><div class="yy-scroll"><table class="yy"><thead><tr><th>指標</th><th>方向</th>
    <th>訓練期間</th><th>成功／成交樣本 ⓘ</th><th>樣本內勝率</th><th>建議限價</th><th>系統篩選</th>
    </tr></thead><tbody>"""]
    for c in sorted(candidates,key=lambda d:(bool(d['達標']),float(d['Wilson 95% 下界%'] or 0)),reverse=True):
        records=c.get("成交樣本明細",pd.DataFrame())
        tip=escape(dates_tooltip(records,max_chars=12000),quote=True)
        win=int(c['成功次數']); total=int(c['符合限價樣本數'])
        r=c['限價成交後勝率%']; rate=f"{r:.1f}%" if np.isfinite(r) else "—"
        limit=format_money(c['歷史估計限價'])
        tag="通過" if c['達標'] else ("樣本不足" if total<cutoff else "未達條件")
        rows.append('<tr>'+f'<td>{escape(c["技術指標"])}</td>'+
                    f'<td>{escape(c["方向"])}</td><td>{escape(c["回測期間"])}</td>'+
                    f'<td><span class="yy-tip" title="{tip}">{win}／{total}</span></td>'+
                    f'<td>{rate}</td><td>{escape(limit)}</td><td>{tag}</td></tr>')
    rows.append('</tbody></table></div><small>ⓘ 滑鼠停在「成功／成交樣本」數字上，可查看每筆訊號日、結果日、成功或失敗；下方可查看全部紀錄。</small>')
    return ''.join(rows)


def stable_checks(best, candidates, threshold, min_samples):
    """把原本三個難懂的研究參數改成三項白話、自動檢查結果。"""
    if best is None:
        has_any=bool(candidates)
        return (
            '近期訊號：尚無通過風控的候選' if has_any else '近期訊號：今日沒有新指標事件',
            '資料是否夠多：尚無足以發出建議的有效樣本',
            '多空意見：沒有通過門檻的可用買賣訊號',
        )
    recent=best['近期加權勝率%']
    eff=best['近期有效樣本數']
    recent_ok=(np.isfinite(recent) and recent>threshold and eff>=RECENT_MIN_EFFECTIVE_N)
    recent_str=(f"{'✅ 最近表現仍達標' if recent_ok else '⚠️ 最近表現不穩定'}："
                f"近期加權勝率 {recent:.1f}%（系統自動檢查）")
    n=int(best['符合限價樣本數']); k=int(best['成功次數'])
    sample_str=(f"{'✅ 證據量通過' if n>=min_samples else '⚠️ 樣本偏少'}："
                f"歷史符合進場價 {n} 筆，其中成功 {k} 筆、失敗 {n-k} 筆；可查看日期")
    sides={r['方向'] for r in candidates if r['達標']}
    conflict='⚠️ 多空有分歧：另有反向候選通過，系統已標示警示' if len(sides)>1 else '✅ 多空沒有達標的反方向衝突'
    return recent_str,sample_str,conflict


def date_status(p:pd.DataFrame):
    now=datetime.now(ZoneInfo('Asia/Taipei'))
    actual=p.index[-1].date()
    # 收盤前，不能把尚未完成的最新日K當作依據。
    if now.weekday()<5 and now.time()<time(14,30) and actual==now.date():
        p=p.loc[p.index.date<now.date()].copy()
        if p.empty: raise ValueError('沒有已完成的日 K')
        actual=p.index[-1].date()
    if now.weekday()<5 and now.time()<time(14,30):
        fresh=False  # 油洗洗只在臺股收盤後產生新一交易日的結論
    elif now.weekday()<5:
        fresh=(actual==now.date())
    else:
        fresh=True
    reason=(f"最後完整 K 線為 {actual}；今日尚未收盤，或行情來源尚未提供今日已完成收盤價；"
            "暫不發布新的隔日訊號。" if not fresh else "")
    return p,fresh,reason


# --- 主體：左側只保留兩個決策門檻，所有分析窗／指標／預測日都自動 ---
st.title('📈 油洗洗｜收盤後的下一交易日研究訊號')
st.caption(f'{VERSION} · GitHub ＋ Streamlit Cloud · 十種指標 × 五個回測期間 · 一日預測 · 最長一年逐日前推驗證')

names, source_errors=stock_directory()
with st.sidebar:
    st.header('股票與兩項判斷條件')
    codes=sorted(names)
    selected=st.selectbox('查詢股票（中文名稱或代號）',codes,
                          index=codes.index('2454') if '2454' in codes else 0,
                          format_func=lambda c:f'{names[c][0]}（{c}｜{names[c][1]}）')
    manual=st.text_input('找不到股票時輸入代號（選填）',placeholder='例如 5347')
    code=manual.strip().upper() or selected
    name,market=find_stock_name(code,names)
    st.divider()
    threshold=st.number_input('① 最低要求的歷史勝率（%）',min_value=0.0,max_value=99.9,
                              value=90.0,step=1.0,
                              help='過去相同訊號、且符合建議限價成交條件的成功比例必須「高於」此值。')
    min_samples=st.number_input('② 最低有效回測樣本數',min_value=1,max_value=200,
                                value=20,step=1,
                                help='分母是符合限價假設、已知道結果的歷史交易，不是只數成功的日子。')
    st.session_state['min_samples']=int(min_samples)
    run=st.button('🔍 更新這檔股票的明日研究建議',type='primary',use_container_width=True)
    st.caption('其餘自動完成：分析今日最新完整收盤、預測下一交易日、十指標、2週／1／3／6／12個月。')
    if source_errors:
        st.warning('股票名單 API 部分失敗，市場掃描可能不完整。')

if run:
    if not re.fullmatch(r'\d{4,6}(?:\.(?:TW|TWO))?',code):
        st.error('股票代號無效；請填 4～6 位數字，或像 5347.TWO。')
        st.stop()
    taipei_today=datetime.now(ZoneInfo('Asia/Taipei')).date()
    start=(pd.Timestamp(taipei_today)-pd.DateOffset(years=3,days=90)).date()
    end=taipei_today+timedelta(days=1)
    try:
        with st.spinner('取得歷史行情、計算十種指標與逐筆證據……'):
            prices, ticker=fetch_daily(code,market,str(start),str(end))
            prices,fresh,stale_reason=date_status(prices)
            values=compute_indicators(prices,DEFAULTS)
            signals=make_signals(values,DEFAULTS)
            ledger=build_ledger(prices,signals,[1],COST_PCT)
            st.session_state['stock_analysis']={'prices':prices,'ticker':ticker,'code':code,
                'name':name,'market':market,'values':values,'signals':signals,
                'ledger':ledger,'fresh':fresh,'stale_reason':stale_reason}
            st.session_state.pop('forward_cache',None)
    except Exception as err:
        st.session_state.pop('stock_analysis',None)
        st.error(f'分析失敗：{err}')

saved=st.session_state.get('stock_analysis')
same=saved is not None and saved.get('code')==code

tab1,tab2,tab3,tab4,tab5=st.tabs([
    '🎯 明日交易結論','🔎 全台股買賣掃描','🧪 逐日前推與樣本證據',
    '📐 十種技術指標公式','🗂️ 原始價格與計算數值'])

forecast=None
forward=None
if same:
    prices=saved['prices']
    forecast=predict_day(prices,saved['signals'],saved['ledger'],len(prices)-1,WINDOW_LABELS,
                         float(threshold),int(min_samples),RECENT_HALFLIFE,RECENT_MIN_EFFECTIVE_N,True)

with tab1:
    if not same:
        st.info('選擇股票後，按左側「更新這檔股票的明日研究建議」。預設會自動比較全部十種指標和五個期間。')
    else:
        st.subheader(f'{saved["name"]}（{code}）｜下一交易日研究計畫')
        last=prices.index[-1].date()
        st.caption(f'收盤資料日：{last}（臺灣時間） · 預測的是下一個實際交易日，不自行猜測休市日 · 已完成日 K')
        verdict=forecast['判斷'] if saved['fresh'] else '觀望'
        model=forecast['主要模型'] if saved['fresh'] else None
        cols=st.columns([1,1,1,1])
        cols[0].metric('今日最後完整收盤價',format_money(forecast['參考收盤價']))
        cols[1].metric('明日方向',verdict)
        cols[2].metric('明日可接受的限價',format_money(model['歷史估計限價']) if model else '無建議價')
        cols[3].metric('限價符合時歷史勝率',f"{model['限價成交後勝率%']:.1f}%" if model else '—')
        if not saved['fresh']:
            st.warning(saved['stale_reason'])
        elif model:
            st.success(f"依 {model['技術指標']} 的 {model['回測期間']} 歷史訊號："
                       f"{model['成功次數']} 筆成功／{model['符合限價樣本數']} 筆符合限價條件；"
                       f"條件歷史勝率 {model['限價成交後勝率%']:.1f}%。")
            st.markdown(f"**估計下一交易日收盤價：{format_money(model['預測次日收盤價'])}**"
                        f"（歷史第 10–90 百分位粗估區間："
                        f"{format_money(model['預測收盤區間10-90%'][0])}～"
                        f"{format_money(model['預測收盤區間10-90%'][1])}）。")
            st.caption('買入＝開盤價不高於買入限價才假設進場；賣出＝看跌方向、開盤價不低於指定限價才假設進場。'
                       '此處的賣出訊號不是已持倉平倉命令，更不代表所有台股都能放空。')
        else:
            st.info('本日沒有通過條件的買入／賣出建議，因此顯示觀望；不會編造價格或勝率。')
        st.markdown('#### 系統自動檢查結果（不用調參數）')
        checks=stable_checks(model,forecast['候選明細'],float(threshold),int(min_samples))
        for line in checks: st.write('• '+line)
        st.caption('這三項由程式自動檢查，不再需要勾選或設定「近期加權有效樣本數、半衰期」等技術參數。')
        st.caption(f'示範往返成本假設：{COST_PCT:.2f}%，不含真實滑價與市場可否放空；近期加權由系統固定管理。')
        with st.expander('為什麼會做出這個判斷？完整理由與其他候選'):
            st.write(forecast['原因'])
            st.dataframe(candidates_frame(forecast['候選明細']).drop(columns=['判讀'],errors='ignore'),
                         use_container_width=True,hide_index=True)

with tab2:
    st.subheader('全台股買入／賣出候選清單')
    st.caption('只顯示達標的買入、賣出；未達標、讀取失敗或尚未掃描的股票不列在候選清單。')
    universe=valid_universe(names)
    scan_key=(str(datetime.now(ZoneInfo('Asia/Taipei')).date()),float(threshold),int(min_samples),len(universe))
    state=st.session_state.get('scanner')
    if state is None or state.get('key') != scan_key:
        state={'key':scan_key,'index':0,'rows':[]}
        st.session_state['scanner']=state
    a,b,c=st.columns(3)
    a.metric('可辨識上市／上櫃股票',len(universe))
    b.metric('已處理股票',state['index'])
    c.metric('有買／賣候選',sum(r.get('建議') in ('買入','賣出') and r.get('狀態')=='成功'
                                for r in state['rows']))
    if len(universe)<100:
        st.warning('目前股票名單過少，疑似官方名單服務未完整回應；這不是全市場結果。')
    now_scan=datetime.now(ZoneInfo('Asia/Taipei'))
    scan_early=now_scan.weekday()<5 and now_scan.time()<time(14,30)
    if scan_early: st.info('目前尚未完成今日收盤，為避免讀到未完成日 K，請於收盤資料更新後掃描。')
    if st.button('▶️ 開始／繼續掃描下一批（每批 40 檔）',
                 disabled=(state['index']>=len(universe) or scan_early)):
        nxt=min(state['index']+40,len(universe))
        now=datetime.now(ZoneInfo('Asia/Taipei'))
        start=(pd.Timestamp(now.date())-pd.DateOffset(years=3,days=90)).date()
        try:
            with st.spinner(f'正在掃描第 {state["index"]+1}～{nxt} 檔，雲端公開報價可能有速率限制……'):
                output=scan_batch(universe[state['index']:nxt],names,str(start),
                         str(now.date()+timedelta(days=1)),pd.Timestamp(now.date()),
                         DEFAULTS,NAMES,float(threshold),int(min_samples),
                         RECENT_HALFLIFE,RECENT_MIN_EFFECTIVE_N,True,COST_PCT)
                state['rows'].extend(output)
                state['index']=nxt
                st.session_state['scanner']=state
        except Exception as e:
            st.error(f'這一批掃描失敗，尚未計入：{e}')
    if universe:
        st.progress(min(1.0,state['index']/len(universe)))
    result=pd.DataFrame(state['rows'])
    if not result.empty:
        if '建議' in result.columns:
            picks=result.loc[(result['狀態']=='成功') & result['建議'].isin(['買入','賣出'])].copy()
        else:
            picks=pd.DataFrame()
        if picks.empty:
            st.info('目前已掃描部分還沒有達標的買入／賣出候選。這不等於全台股沒有機會。')
        else:
            view_cols=['股票代號','股票名稱','市場','訊號日','收盤價','建議','指標',
                       '回測期間','建議限價','價格條件勝率%','有效樣本數']
            picks=picks[view_cols].sort_values(['建議','價格條件勝率%'],ascending=[True,False])
            st.dataframe(picks,use_container_width=True,hide_index=True)
            st.download_button('⬇️ 下載目前買入／賣出候選 CSV',
                picks.to_csv(index=False).encode('utf-8-sig'),
                file_name='youxixi_scan_candidates.csv',mime='text/csv')
        failed=(result['狀態']=='讀取失敗').sum()
        st.caption(f'目前掃描 {state["index"]}/{len(universe)}，其中 {failed} 檔資料不足或取得失敗。'
                   '未掃描股票不是觀望，部分掃描結果也不是全市場排序。')
    else:
        st.info('點上方「開始／繼續掃描」以取得第一批結果。首次掃描不會自動下單。')

with tab3:
    st.subheader('回測一年＝把過去一年每一天當成當時的明天')
    st.caption('例如要驗證 2026/11/04，模型只能使用 2026/11/03（含）以前已完成的資料；'
               '接著驗證 11/03、11/02……直至一年前的交易日。每一天都重新決定指標與訓練期間。')
    if not same:
        st.info('請先於左側選一檔股票並執行分析。')
    else:
        st.write('**目前所有新訊號候選 × 五個期間的歷史樣本（滑鼠移到數字可看到日期）**')
        st.html(hover_candidates_html(forecast['候選明細'],int(min_samples)))
        if forecast['候選明細']:
            labeler=lambda i:f"{i+1}. {forecast['候選明細'][i]['技術指標']}｜{forecast['候選明細'][i]['回測期間']}｜{forecast['候選明細'][i]['方向']}"
            which=st.selectbox('查看哪一組的完整成功／失敗日期？',
                               list(range(len(forecast['候選明細']))),format_func=labeler)
            m=forecast['候選明細'][which]
            fills=m.get('成交樣本明細',pd.DataFrame())
            if fills.empty:
                st.info('該條件沒有已完成的符合限價樣本。')
            else:
                st.caption(f"成功 {int(fills['是否成功'].sum())} 筆、失敗 {int((~fills['是否成功']).sum())} 筆。"
                           '完整列出訊號日、預測日、假設買賣價與結果。')
                st.dataframe(fills.sort_values('訊號日',ascending=False),
                    use_container_width=True,hide_index=True)
                st.download_button('⬇️ 下載這組所有成功／失敗日期',
                    fills.to_csv(index=False).encode('utf-8-sig'),
                    file_name=f'youxixi_{code}_signal_dates.csv',mime='text/csv')
        st.divider()
        st.subheader('逐日前推測試：過去一年，每天只用昨天以前可得的資訊')
        current_key=(code,str(prices.index[-1].date()),float(threshold),int(min_samples))
        cached=st.session_state.get('forward_cache')
        if cached is not None and cached.get('key')==current_key:
            forward=cached['data']
        else:
            with st.spinner('以過去一年每個交易日為目標逐步回測，正在防止未來價格洩漏……'):
                forward=walk_forward(prices,saved['signals'],saved['ledger'],
                    pd.Timestamp(prices.index[-1])-pd.DateOffset(years=1),WINDOW_LABELS,
                    float(threshold),int(min_samples),RECENT_HALFLIFE,
                    RECENT_MIN_EFFECTIVE_N,True,COST_PCT)
                st.session_state['forward_cache']={'key':current_key,'data':forward}
        if forward.empty:
            st.warning('無法取得完整的一年歷史逐日驗證。')
        else:
            buysell=forward.loc[forward['判斷'].isin(['買入','賣出'])]
            fills=buysell.loc[buysell['限價符合開盤條件']]
            a,b,c,d=st.columns(4)
            a.metric('已驗證交易日',len(forward))
            b.metric('發出買／賣候選天數',len(buysell))
            c.metric('開盤符合限價天數',len(fills))
            d.metric('符合限價後成功率',f"{100*(fills['依開盤成交假設扣成本報酬%']>0).mean():.1f}%" if len(fills) else '樣本不足')
            st.dataframe(forward.sort_values('預測目標日',ascending=False),
                         use_container_width=True,hide_index=True)
            st.download_button('⬇️ 下載全部逐日前推明細 CSV',
                   forward.to_csv(index=False).encode('utf-8-sig'),
                   file_name=f'youxixi_{code}_walk_forward_1year.csv',mime='text/csv')
            st.caption('前推驗證與同日的「樣本內勝率」是不同指標。成功率僅用符合開盤限價假設且扣示範成本後仍獲利的交易計算；'
                       '開盤不符合價格的不當成虧損或成功。')

with tab4:
    st.subheader('十種技術指標計算公式與買賣規則')
    st.caption('一目均衡表不使用未來雲層數值；所有訊號收盤後才能確認。公式頁永久可用，不需先下載行情。')
    for n,(title,label,formula,url) in enumerate(formula_sections(DEFAULTS),start=1):
        with st.expander(f'{n}. {title}｜{label}'):
            st.markdown(formula)
            st.markdown(f'[外部參考資料]({url})')
    with st.expander('顯示指標與訊號程式原始碼'):
        st.code(inspect.getsource(compute_indicators),language='python')
        st.code(inspect.getsource(make_signals),language='python')
    with st.expander('回測的價格及勝率定義'):
        st.markdown('**訊號日 T**：只能用當天收盤以前的資料；**進場日 T+1**：開盤價符合建議限價才假設進場；'
                    '**平倉日 T+1**：當日收盤價；看跌是反向報酬代理，不代表合法放空已驗證。')
        st.latex(r'P_{win}=\frac{\text{符合限價且扣成本後獲利次數}}{\text{符合限價且結果已完成的樣本次數}}\times100\%')
        st.markdown('限價取自當時訓練窗的歷史次日開盤跳空幅度分位數，再依臺股一般股票跳動單位保守取整。'
                    '預估收盤價取相同類型訊號下一日收盤變動的歷史中位數，並非「未來收盤一定會到此價」。')
        st.markdown(f'近期加權由系統固定採 **{RECENT_HALFLIFE} 個交易日半衰期**，'
                    f'近期有效樣本最低 **{RECENT_MIN_EFFECTIVE_N}**；這是自動風控，不放在左側要求你調整。'
                    '逐日前推的每一天只能用當時已完成的交易紀錄重新訓練。')

with tab5:
    if not same:
        st.info('執行單股分析後，此處顯示原始日 K 與所有公式的中間數值。')
    else:
        st.subheader('原始日 K、所有技術指標與逐日事件')
        frame=saved['prices'].join(saved['values']).copy()
        for nm,series in saved['signals'].items(): frame['訊號｜'+nm]=series
        frame.index.name='交易日'
        st.dataframe(frame.tail(60),use_container_width=True)
        st.download_button('⬇️ 匯出完整計算 CSV',
                           frame.to_csv().encode('utf-8-sig'),
                           file_name=f'youxixi_{code}_indicator_audit.csv',mime='text/csv')

st.divider()
with st.expander('⚠️ 研究限制與使用前必讀'):
    st.markdown('''- 本系統是**收盤後的下一交易日研究與模擬**，不是即時當沖建議或自動下單；預測價是統計估計，未經真實成交驗證。\n- Yahoo Finance 行情可能延遲、缺漏、調整或限流。\n- 90% 是**歷史符合進場條件的樣本內成功比例**，不是已校準的明日機率；同時比較十個指標和五個重疊窗口，仍有多重比較與選模偏誤。\n- 過去一年逐日前推結果才較接近真正的樣本外驗證，但同樣不能保證未來，且資料可能含事後除權息調整。\n- 所有交易成本使用示範固定 0.45%，沒有足夠盤中資料檢驗 9 點至收盤間的限價是否成交、停利停損先後、流動性、滑價、借券限制與真實稅費。\n- 全市場掃描為分批執行，公開資料速率限制可能導致缺漏；掃描進度未達 100% 時，不應當成全台股結論。\n- 選擇權模組暫不納入。本版的「賣出」為看跌方向，不等於已確認可以實際放空。''')

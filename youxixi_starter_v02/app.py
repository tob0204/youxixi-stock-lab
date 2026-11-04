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
from scan_engine import scan_batch
from sector_engine import load_official_industries, sectors_summary, INDUSTRIES

VERSION = "v0.9"
COST_PCT = 0.45  # 示範用，非券商實際費率；均為報酬率百分點。
RECENT_HALFLIFE = 20  # 開發者自動風控，不要求使用者調整。
RECENT_MIN_EFFECTIVE_N = 3
HORIZON = 1
# 自動證據篩選：不暴露勝率門檻給使用者，避免不斷調到恰好出現訊號。
AUTO_BASELINE_WIN_RATE = 50.0
AUTO_MIN_FILLED_SAMPLES = 8
MAX_SCAN_PER_ACTION = 100
SCAN_BATCH_SIZE = 20
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
            "判讀":"候選達標" if m["達標"] else (
                "樣本不足" if m["符合限價樣本數"] < AUTO_MIN_FILLED_SAMPLES else
                "回測正期望值或近期穩定性不足"),
        })
    return pd.DataFrame(rows)


def hover_candidates_html(candidates:list[dict], min_samples=8):
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
        tag="候選" if c['達標'] else ("樣本不足" if total<min_samples else "證據不足")
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



# --- 畫面：股票搜尋 + 指定產業海選；不再出現其他技術調參欄位 ---
@st.cache_data(ttl=12*3600, show_spinner=False)
def sector_directory():
    return load_official_industries()

@st.cache_data(ttl=3600, show_spinner=False)
def scan_sector_chunk(codes, names_for_sector, start, end, now_date):
    # 每一批做快取；切換 Streamlit 頁籤不重抓相同一批。
    return scan_batch(list(codes), names_for_sector, start, end, pd.Timestamp(now_date),
                      DEFAULTS, NAMES, AUTO_BASELINE_WIN_RATE, AUTO_MIN_FILLED_SAMPLES,
                      RECENT_HALFLIFE, RECENT_MIN_EFFECTIVE_N, True, COST_PCT)

st.title('📈 油洗洗｜收盤後預測下一個交易日')
st.caption(f'{VERSION} · GitHub + Streamlit 網頁版 · 十指標 × 2週／1／3／6／12個月自動比較 · 只預測下一交易日')
names, source_errors=stock_directory()
with st.sidebar:
    st.header('股票查詢')
    codes=sorted(names)
    selected=st.selectbox('股票代號／中文名稱',codes,
                          index=codes.index('2454') if '2454' in codes else 0,
                          format_func=lambda c:f'{names[c][0]}（{c}｜{names[c][1]}）')
    manual=st.text_input('查無股票時，可直接輸入代號（選填）')
    code=manual.strip().upper() or selected
    name,market=find_stock_name(code,names)
    run=st.button('📊 分析這檔股票',type='primary',use_container_width=True)
    st.divider()
    st.caption('其他設定交給系統：當日最後完整收盤、十種技術指標、五種歷史窗、下一交易日、近期驗證。')
    if source_errors:
        with st.expander('中文名稱來源狀態'):
            for error in source_errors: st.warning(error)

if run:
    if not re.fullmatch(r'\d{4,6}(?:\.(?:TW|TWO))?',code):
        st.error('股票代號格式錯誤，請輸入 4～6 位數字，例如 2454。')
        st.stop()
    now=datetime.now(ZoneInfo('Asia/Taipei'))
    start=(pd.Timestamp(now.date())-pd.DateOffset(years=3,days=90)).date()
    try:
        with st.spinner('計算技術指標與每筆歷史交易……'):
            prices,ticker=fetch_daily(code,market,str(start),str(now.date()+timedelta(days=1)))
            prices,fresh,stale_reason=date_status(prices)
            values=compute_indicators(prices,DEFAULTS)
            signals=make_signals(values,DEFAULTS)
            ledger=build_ledger(prices,signals,[1],COST_PCT)
            st.session_state['single']={'prices':prices,'ticker':ticker,'code':code,
                'name':name,'market':market,'values':values,'signals':signals,
                'ledger':ledger,'fresh':fresh,'stale_reason':stale_reason}
            st.session_state.pop('forward_cache_v09',None)
    except Exception as exc:
        st.session_state.pop('single',None)
        st.error(f'單股分析失敗：{exc}')

saved=st.session_state.get('single')
same=saved is not None and saved.get('code')==code
forecast=None
if same:
    prices=saved['prices']
    try:
        forecast=predict_day(prices,saved['signals'],saved['ledger'],len(prices)-1,
                             WINDOW_LABELS,AUTO_BASELINE_WIN_RATE,AUTO_MIN_FILLED_SAMPLES,
                             RECENT_HALFLIFE,RECENT_MIN_EFFECTIVE_N,True)
    except Exception as exc:
        st.error(f'回測引擎出現異常：{exc}。可至 Streamlit 日誌取得詳細追蹤。')

tab1,tab2,tab3,tab4,tab5=st.tabs([
    '🎯 明日買賣訊號','🏭 依產業海選','🔎 逐日前推與樣本日期',
    '📐 十種指標公式','🗂️ 原始指標數值'])

with tab1:
    st.subheader('個股：收盤後對下一交易日的預測')
    if not same or forecast is None:
        st.info('請從左側選股票並按「分析這檔股票」。')
    else:
        best=forecast['主要模型'] if saved['fresh'] else None
        verdict=forecast['判斷'] if saved['fresh'] else '觀望'
        st.markdown(f'#### {name}（{code}）｜最後完整收盤日 {prices.index[-1]:%Y-%m-%d}')
        if not saved['fresh']:
            st.warning('尚未取得今天完成的收盤資料：'+saved['stale_reason'])
        c1,c2,c3,c4=st.columns(4)
        c1.metric('最新收盤價',format_money(forecast['參考收盤價']))
        c2.metric('下一交易日方向',verdict)
        c3.metric('歷史模型建議限價',format_money(best['歷史估計限價']) if best else '無')
        c4.metric('該限價條件樣本內勝率',f"{best['限價成交後勝率%']:.1f}%" if best else '—')
        if best:
            st.success(f"{best['技術指標']}｜{best['回測期間']}："
                       f"成功 {best['成功次數']} 次／符合限價 {best['符合限價樣本數']} 次；"
                       f"平均扣示範成本報酬 {best['平均扣成本報酬%']:.2f}%。")
            st.write('**預測下一日收盤參考值：**'+format_money(best['預測次日收盤價']))
            st.caption('建議限價是歷史開盤跳空分布推估，不是保證成交價格；收盤預測為樣本內統計估計。')
        else:
            st.info(f"系統本次判定：**觀望**。{forecast['原因']}")
            if forecast['研究性候選']:
                p=forecast['研究性候選']
                st.caption(f"仍可核對未達標候選：{p['技術指標']}／{p['回測期間']}，"
                           f"樣本 {p['符合限價樣本數']}，目前不作買賣推薦。")
        with st.expander('為什麼是這個結果？其他所有指標與期間'):
            st.write(forecast['原因'])
            st.dataframe(candidates_frame(forecast['候選明細']),
                         use_container_width=True,hide_index=True)
        st.caption(f'自動採用基本證據標準：至少 {AUTO_MIN_FILLED_SAMPLES} 筆歷史符合限價樣本、'
                   '扣成本後平均報酬為正、近期表現未明顯失效；不讓使用者反覆調勝率門檻以製造訊號。'
                   '本版只有日 K，不能證明實際盤中委託與放空可成交。')

with tab2:
    st.subheader('依產業海選｜只顯示下一交易日買入／賣出候選')
    st.caption('先選一個官方產業，按一次「掃描此產業」；不會掃其他產業。交易對象限上市／上櫃四碼股票。')
    profiles,sources,profile_errors=sector_directory()
    if profile_errors:
        for err in profile_errors: st.warning(err)
    if not profiles:
        st.error('官方公司產業分類目前無法取得。為避免錯誤分組，暫停海選。')
    else:
        categories=sectors_summary(profiles)
        ids=sorted(categories.keys())
        chosen=st.selectbox('選擇產業',ids,
             format_func=lambda i:f'{INDUSTRIES.get(i,i)}（{len(categories[i])} 檔；{i}）')
        selected_codes=categories[chosen]
        st.caption(f'資料來源：{"、".join(sources)}公司基本資料，產業「{INDUSTRIES[chosen]}」，'
                   f'共 {len(selected_codes)} 檔。沒有分類資料的個股不擅自加入。')
        now=datetime.now(ZoneInfo('Asia/Taipei'))
        # 股票資料缺日仍不可用最近幾天的訊號冒充已核對今日收盤的結論。
        day_key=now.date().isoformat()
        scan_key=(chosen,day_key,tuple(selected_codes))
        all_states=st.session_state.setdefault('sector_scans_v09',{})
        state=all_states.get(chosen)
        if not state or state.get('key')!=scan_key:
            state={'key':scan_key,'next':0,'rows':[]}
            all_states[chosen]=state
        scan_early=now.weekday()<5 and now.time()<time(14,30)
        if scan_early:
            st.info('台股今日尚未收盤，請等待完整收盤資料更新後再進行下一交易日海選。')
        btn,reset_col=st.columns([4,1])
        scan_pressed=btn.button('🔍 掃描此產業的明日候選',type='primary',use_container_width=True,
                                disabled=(state['next']>=len(selected_codes) or scan_early))
        reset_pressed=reset_col.button('重置本產業結果',use_container_width=True)
        if reset_pressed:
            state={'key':scan_key,'next':0,'rows':[]}
            all_states[chosen]=state
            st.rerun()
        if scan_pressed:
            start=(pd.Timestamp(now.date())-pd.DateOffset(years=3,days=90)).date()
            end=now.date()+timedelta(days=1)
            names_in_sector={c:(profiles[c]['名稱'],profiles[c]['市場']) for c in selected_codes}
            finish=min(len(selected_codes),state['next']+MAX_SCAN_PER_ACTION)
            progress=st.progress(state['next']/len(selected_codes))
            note=st.empty()
            while state['next']<finish:
                chunk=selected_codes[state['next']:min(finish,state['next']+SCAN_BATCH_SIZE)]
                note.info(f'正在掃描 {INDUSTRIES[chosen]}：已完成 {state["next"]}／{len(selected_codes)} 檔')
                try:
                    rows=scan_sector_chunk(tuple(chunk),names_in_sector,str(start),str(end),str(now.date()))
                except Exception as exc:
                    rows=[{'股票代號':c,'股票名稱':names_in_sector[c][0],
                           '狀態':'讀取失敗','原因':str(exc)[:150]} for c in chunk]
                state['rows'].extend(rows)
                state['next']+=len(chunk)
                progress.progress(state['next']/len(selected_codes))
            all_states[chosen]=state
            note.empty()
        done=state['next']
        data=pd.DataFrame(state['rows'])
        valid=(data.loc[(data['狀態']=='成功') & data['建議'].isin(['買入','賣出'])]
               if not data.empty and {'狀態','建議'}.issubset(data.columns) else pd.DataFrame())
        a,b,c=st.columns(3)
        a.metric('此產業股票數',len(selected_codes))
        b.metric('已掃描／總數',f'{done}／{len(selected_codes)}')
        c.metric('買賣候選',len(valid))
        if done<len(selected_codes):
            st.info('此產業尚未全部掃描，結果是部分清單。再次按「掃描此產業」只會接續本產業未掃描的股票。'
                    '為降低免費雲端服務被公開報價限流的風險，每次最多處理 100 檔。')
        else:
            st.success('此產業的本次掃描已完成（有資料讀取失敗者仍會另外計數）。')
        if not valid.empty:
            keys=['股票代號','股票名稱','市場','訊號日','收盤價','建議',
                  '指標','回測期間','建議限價','價格條件勝率%','有效樣本數']
            valid=valid[keys].sort_values(['建議','價格條件勝率%'],ascending=[True,False])
            st.dataframe(valid,hide_index=True,use_container_width=True)
            st.download_button('下載本產業買賣候選 CSV',valid.to_csv(index=False).encode('utf-8-sig'),
                               file_name=f'youxixi_{chosen}_candidates.csv',mime='text/csv')
        elif done:
            st.info('目前已完成部分尚無符合自動證據檢查的買入／賣出候選；觀望股票不列入表格。')
        else:
            st.info('請選擇產業後按「掃描此產業」，開始尋找買入／賣出候選。')
        if not data.empty and '狀態' in data.columns:
            fails=data[data['狀態']!='成功']
            st.caption(f'已處理 {done} 檔，其中 {len(fails)} 檔因缺資料或報價服務限制無法完成。')
            if not fails.empty:
                with st.expander(f'查看 {len(fails)} 檔失敗股票與原因'):
                    st.dataframe(fails[['股票代號','股票名稱','原因']],hide_index=True,use_container_width=True)
        st.caption('「賣出」是預測偏跌的研究訊號，不代表該股具備合法或實際可執行的現股當沖放空條件。'
                   '掃描使用公開歷史日 K，報價來源可能限速、更新延遲或缺漏。')

with tab3:
    st.subheader('逐日往前模擬：每一天只能使用前一交易日已完成的資訊')
    if not same or forecast is None:
        st.info('請先在左側分析一檔股票。')
    else:
        st.markdown('**最新訊號歷史樣本：滑鼠移到「成功／樣本」可查看日期**')
        st.html(hover_candidates_html(forecast['候選明細'],AUTO_MIN_FILLED_SAMPLES))
        candidates=forecast['候選明細']
        if candidates:
            idx=st.selectbox('完整檢查一組指標的成功／失敗日子',range(len(candidates)),
                 format_func=lambda i:f"{candidates[i]['技術指標']}｜{candidates[i]['回測期間']}｜{candidates[i]['方向']}")
            m=candidates[idx]
            dates=m.get('成交樣本明細',pd.DataFrame())
            if not dates.empty:
                st.dataframe(dates.sort_values('訊號日',ascending=False),hide_index=True,use_container_width=True)
                st.download_button('匯出這組完整樣本 CSV',dates.to_csv(index=False).encode('utf-8-sig'),
                    file_name=f'youxixi_{code}_dates.csv',mime='text/csv')
            else: st.info('這組沒有符合進場價格的已完成樣本。')
        st.divider()
        st.write('**往前一年，逐日重新訓練並驗證下一交易日的結果**')
        key=(code,str(prices.index[-1].date()),VERSION)
        cached=st.session_state.get('forward_cache_v09')
        if st.button('▶️ 計算過去一年逐日前推回測',type='secondary'):
            with st.spinner('逐日回測（僅用每個歷史日期當時已知的資料）……'):
                try:
                    output=walk_forward(prices,saved['signals'],saved['ledger'],
                        pd.Timestamp(prices.index[-1])-pd.DateOffset(years=1),WINDOW_LABELS,
                        AUTO_BASELINE_WIN_RATE,AUTO_MIN_FILLED_SAMPLES,
                        RECENT_HALFLIFE,RECENT_MIN_EFFECTIVE_N,True,COST_PCT)
                    st.session_state['forward_cache_v09']={'key':key,'data':output}
                except Exception as exc:
                    st.error(f'前推回測失敗：{exc}')
        cached=st.session_state.get('forward_cache_v09')
        if cached is not None and cached['key']==key:
            out=cached['data']
            if out.empty: st.info('沒有足夠歷史資料供前推測試。')
            else:
                sent=out[out['判斷'].isin(['買入','賣出'])]
                filled=sent[sent['限價符合開盤條件']]
                x,y,z=st.columns(3)
                x.metric('驗證日期',len(out))
                y.metric('發出買／賣訊號',len(sent))
                z.metric('符合假設限價後勝率',f"{100*(filled['依開盤成交假設扣成本報酬%']>0).mean():.1f}%" if len(filled) else '不足')
                st.dataframe(out.sort_values('預測目標日',ascending=False),hide_index=True,use_container_width=True)
                st.download_button('匯出逐日前推 CSV',out.to_csv(index=False).encode('utf-8-sig'),
                                   file_name=f'youxixi_{code}_walk_forward.csv',mime='text/csv')
        st.caption('逐日前推勝率與事後對過去某段期間的樣本內勝率不同。逐日前推才是比較接近實際驗證的統計。')

with tab4:
    st.subheader('十種技術指標計算公式、參數與訊號規則')
    for n,(title,label,formula,url) in enumerate(formula_sections(DEFAULTS),start=1):
        with st.expander(f'{n}. {title}｜{label}'):
            st.markdown(formula)
            st.markdown(f'[外部參考資料]({url})')
    with st.expander('查看系統實際計算程式碼'):
        st.code(inspect.getsource(compute_indicators),language='python')
        st.code(inspect.getsource(make_signals),language='python')

with tab5:
    if not same:
        st.info('分析個股後可看到每一天價格與指標中間數值。')
    else:
        data=saved['prices'].join(saved['values'])
        for indicator,series in saved['signals'].items():
            data['訊號｜'+indicator]=series
        st.dataframe(data.tail(60),use_container_width=True)
        st.download_button('匯出完整價格／指標 CSV',data.to_csv().encode('utf-8-sig'),
                           file_name=f'youxixi_{code}_formula_audit.csv',mime='text/csv')

st.divider()
with st.expander('⚠️ 模型與實際交易限制'):
    st.markdown("""- 日 K 模型：今天收盤後分析，假設下一交易日開盤價格滿足限價即成交、當日收盤平倉。**目前不是分鐘 K 或真實盤中成交回測**。
- 系統自動挑選五種歷史窗口與十個指標，仍可能出現事後挑選最佳值與多重比較偏誤；樣本內成功率不是明日真實勝率。
- 逐日前推每一天只能使用當時完成的歷史資料，但目前股票名單未做歷史成分股重建，產業海選仍有倖存者偏誤。
- 預設成本 0.45% 僅是示例，不代表券商真實手續費、交易稅、融券成本或滑價。
- 「賣出」是看跌方向統計代理，**不等於可以合法或實務上立即放空**；要真正當沖必須核對個股當沖資格、交易制度與報價流動性。
- 若當天尚未有完整日 K、官方產業 API 回覆不完整或行情來源限流，不會把舊資料／未掃描股票冒充最新有效研究結果。
- 選擇權尚未納入。""")

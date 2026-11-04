"""油洗洗 v0.6｜台股十種技術指標 × 日線歷史回測 × 公式透明核對。

GitHub + Streamlit Community Cloud；無需在使用者本機安裝。
限制：日線方向研究原型、非當沖或選擇權績效、無券商自動下單。
"""
from __future__ import annotations

from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
import inspect
import re

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from indicators import DEFAULTS, NAMES, compute_indicators, make_signals, validate_params
from formulas import formula_sections
from signal_engine import evaluate, decide_latest
from stock_names import fetch_stock_names, find_stock_name

HORIZONS = (1, 3, 5, 10, 20)
VERSION = "v0.6"
st.set_page_config(page_title="油洗洗｜十種技術指標回測", page_icon="📈", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(stock_id: str, market: str, start: str, end_exclusive: str):
    """Yahoo Finance 日線，自動調整 OHLC，end 為不包含上界。"""
    if "." in stock_id:
        symbols = [stock_id]
    else:
        symbols = ([f"{stock_id}.TWO", f"{stock_id}.TW"] if market == "上櫃"
                   else [f"{stock_id}.TW", f"{stock_id}.TWO"])
    errors = []
    for symbol in symbols:
        try:
            df = yf.Ticker(symbol).history(
                start=start, end=end_exclusive, interval="1d", auto_adjust=True,
                actions=False, timeout=20,
            )
            if df is None or df.empty:
                errors.append(f"{symbol}：無資料")
                continue
            cols = {"Open", "High", "Low", "Close", "Volume"}
            if not cols.issubset(df.columns):
                errors.append(f"{symbol}：OHLCV 欄位不完整")
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            idx = pd.DatetimeIndex(pd.to_datetime(df.index))
            if idx.tz is not None:
                idx = idx.tz_convert("Asia/Taipei").tz_localize(None)
            df.index = idx.normalize()
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Open", "High", "Low", "Close"])
            df = df[(df["Close"] > 0) & (df["Open"] > 0)]
            if len(df) < 100:
                errors.append(f"{symbol}：日線少於 100 筆")
                continue
            return df, symbol
        except Exception as exc:
            errors.append(f"{symbol}：{str(exc)[:130]}")
    raise ValueError("無法取得足夠的日線行情。" + "；".join(errors))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_stock_names():
    return fetch_stock_names()


# -------- sidebar and research input --------
st.title("📈 油洗洗｜台股量化技術指標回測")
st.caption(f"{VERSION} · GitHub × Streamlit · 10 種指標與完整計算公式｜目前使用日線，**尚非當沖回測**")

with st.sidebar:
    st.header("研究條件")
    names, errors = cached_stock_names()
    codes = sorted(names)
    selected_code = st.selectbox(
        "股票（中文名稱或代號搜尋）", codes,
        index=codes.index("2454") if "2454" in codes else 0,
        format_func=lambda c: f"{names[c][0]}（{c}｜{names[c][1]}）",
        help="選單可輸入股票代號或中文名稱；名稱來源優先採證交所／櫃買中心開放資料。",
    )
    with st.expander("查無股票？自行輸入代號"):
        manual_code = st.text_input("手動股票代號（選填）", placeholder="例如：6488.TWO")
    code = manual_code.strip().upper() or selected_code
    name, market = find_stock_name(code, names)
    st.caption(f"目前研究：**{name}（{code}）**｜{market}")
    if errors:
        st.warning("部分官方股票名稱服務暫時無法使用；可能只顯示備援名單。", icon="⚠️")

    today = datetime.now(ZoneInfo("Asia/Taipei")).date()
    asof = st.date_input("歷史分析基準日", value=today, max_value=today)
    months = st.selectbox("往前回測", [1, 3, 6, 12, 24, 36], index=3,
                          format_func=lambda n: f"{n} 個月")
    horizons = st.multiselect("預測期限（交易日）", list(HORIZONS), default=list(HORIZONS))
    chosen = st.multiselect("本次參與回測的指標", NAMES, default=NAMES)
    st.divider()
    st.subheader("🎯 最新訊號條件")
    success_threshold = st.number_input(
        "買入／賣出門檻：歷史成功率必須高於 (%)", min_value=0.0,
        max_value=100.0, value=90.0, step=1.0,
        help="例如設 90%，必須嚴格大於 90% 才符合，不是等於。使用的是假設成本後歷史方向成功率，非校準的未來機率。",
    )
    min_n = st.number_input("歷史最低樣本數", min_value=1, max_value=500, value=20,
                             help="只有當前有新訊號、該訊號歷史完成樣本數達門檻，才可能顯示買入或賣出。")
    require_recent = st.checkbox("同時要求近期加權成功率超過門檻（建議）", value=True,
                                 help="歷史整體勝率高但近期效果失效時不觸發買賣。")
    effective_n_min = st.number_input("近期最低有效樣本數", min_value=1.0, max_value=100.0,
                                      value=5.0, step=1.0,
                                      help="只在啟用近期加權門檻時使用，有效樣本數會因衰減而小於原始筆數。")
    half_life = st.number_input("近期加權半衰期（交易日）", min_value=1, max_value=250, value=20,
                                help="依訊號結果完成的日期加權；每隔指定交易日權重減半。")
    round_trip_cost_pct = st.number_input("假設往返成本合計（%）", min_value=0.0,
                                         max_value=5.0, value=0.45, step=0.05,
                                         help="預設 0.45% 僅為示範估計值，請依你的券商、稅費與滑價修改；這不是實際交易精算。")
    params: dict[str, int | float] = dict(DEFAULTS)
    with st.expander("⚙️ 指標參數（可修改）", expanded=False):
        st.caption("所有參數為 K 線根數；不填入任何未來價格。頁籤會同步顯示使用的參數與公式。")
        labels = {
            "ma_fast": "MA 短", "ma_slow": "MA 長", "ema_fast": "EMA 短", "ema_slow": "EMA 長",
            "kd_window": "KD RSV", "kd_smooth": "KD 平滑 m", "macd_fast": "MACD 快",
            "macd_slow": "MACD 慢", "macd_signal": "MACD 訊號",
            "bb_window": "布林期間", "bb_k": "布林標準差倍數 k", "rsi_window": "RSI 期間",
            "fib_window": "斐波那契滾動區間", "ichi_conversion": "一目轉換線",
            "ichi_base": "一目基準線", "ichi_span_b": "一目先行 B",
            "ichi_shift": "一目雲位移", "std_window": "標準差期間", "std_z": "Z 分數門檻",
            "adx_window": "ADX 期間", "adx_threshold": "ADX 強度門檻",
        }
        for key, default in DEFAULTS.items():
            if isinstance(default, float):
                params[key] = st.number_input(labels[key], min_value=0.1, max_value=100.0,
                                              value=default, step=0.1, key=f"param_{key}")
            else:
                params[key] = int(st.number_input(labels[key], min_value=2 if key != "kd_smooth" else 1,
                                                  max_value=250, value=default, step=1,
                                                  key=f"param_{key}"))
    run = st.button("開始回測", type="primary", use_container_width=True)

condition = (code, str(asof), int(months), tuple(horizons), tuple(chosen),
             int(half_life), float(round_trip_cost_pct), tuple(sorted(params.items())))

if run:
    if not re.fullmatch(r"\d{4,6}(?:\.(?:TW|TWO))?", code):
        st.error("股票代號格式不正確，例如 2454、2330.TW 或 6488.TWO。")
        st.stop()
    if not chosen or not horizons:
        st.warning("至少選擇一個技術指標及一個預測期限。")
        st.stop()
    try:
        validate_params(params)
    except ValueError as exc:
        st.error(f"參數不合理：{exc}")
        st.stop()

    start = pd.Timestamp(asof) - pd.DateOffset(months=months)
    # 指標暖機：讓 Wilder RSI/ADX 和長 EMA 有超過 250 根歷史價格可使用。
    fetch_start = start - pd.Timedelta(days=540)
    fetch_end = pd.Timestamp(asof) + timedelta(days=1)
    with st.spinner("正在下載行情並核對 10 種技術指標……"):
        try:
            prices, symbol = load_prices(code, market, str(fetch_start.date()), str(fetch_end.date()))
            # 今日尚未完成的日 K 不應當成已知收盤後訊號；臺股以 14:30 保守當作完成界線。
            if asof == today and datetime.now(ZoneInfo("Asia/Taipei")).time() < time(14, 30):
                prices = prices.loc[prices.index < pd.Timestamp(today)].copy()
            prices = prices.loc[prices.index <= pd.Timestamp(asof)].copy()
            if prices.empty or prices.index.max() < start:
                raise ValueError("所選回測區間沒有足夠資料。")
            values = compute_indicators(prices, params)
            all_signals = make_signals(values, params)
            active_signals = {key: all_signals[key] for key in chosen}
            result = evaluate(prices, active_signals, start, horizons,
                              int(half_life), float(round_trip_cost_pct))
        except Exception as exc:
            st.session_state.pop("youxixi_data", None)
            st.error(f"取得資料或計算失敗：{exc}")
            st.stop()
    st.session_state["youxixi_data"] = {
        "condition": condition, "prices": prices, "values": values,
        "signals": active_signals, "result": result, "symbol": symbol,
        "code": code, "name": name, "asof": str(asof), "params": params,
    }

# 90% 等決策控制值不屬於行情快取條件；調整後立即重新判斷，不必重跑行情。
# tabs are always available, and formulas work even before any API request succeeds.
tab_signal, tab_results, tab_formulas, tab_verify = st.tabs([
    "🎯 最新交易訊號", "📊 歷史回測結果", "📐 十種指標計算公式", "🔍 原始數據與逐日核對",
])

saved = st.session_state.get("youxixi_data")
up_to_date = saved is not None and saved.get("condition") == condition

with tab_signal:
    st.subheader("最新買入／賣出／觀望判斷")
    if not up_to_date:
        st.info("請選好研究條件並點選左側『開始回測』。完成後，調整勝率門檻會即時更新判斷。")
    else:
        prices, signals, results = saved["prices"], saved["signals"], saved["result"]
        signal_day = prices.index[-1].date()
        st.caption(f"{name}（{code}） · 最新資料交易日：{signal_day} · 日線收盤後策略 · 不是真正的盤中即時報價")
        st.metric("最近有效日 K 調整後收盤價（非即時報價）", f"NT$ {prices['Close'].iloc[-1]:,.2f}")
        if not horizons:
            st.warning("請先選擇至少一個預測期限。")
        else:
            selected_horizon = st.selectbox("這次訊號要預測幾個交易日後？", sorted(horizons),
                                             index=0, format_func=lambda x: f"{x} 個交易日")
            historical_mode = asof != today
            # 不將資料滯後誤稱為即時交易建議：超過 2 日的非週末交易日差需要提示。
            # 假日／休市也可能發生，因此一律清楚顯示實際日期。
            stale_today = (not historical_mode and today.weekday() < 5 and
                           datetime.now(ZoneInfo("Asia/Taipei")).time() >= time(14, 30)
                           and signal_day != today)
            allowed = not historical_mode and not stale_today
            verdict, signal_table, explanation = decide_latest(
                signals, results, int(selected_horizon), float(success_threshold),
                int(min_n), float(effective_n_min), bool(require_recent), allow_signal=allowed,
            )
            if historical_mode:
                verdict, explanation = "歷史模擬", f"你選擇的基準日是 {asof}，不能稱為當下交易建議。" + explanation
            elif stale_today:
                verdict, explanation = "觀望", "今日已過保守收盤確認時間，但行情來源尚未提供今日完整日 K；禁止沿用舊訊號。"
            color = {"買入": "green", "賣出": "red", "觀望": "gray", "歷史模擬": "blue"}.get(verdict, "gray")
            st.markdown(f"### 訊號結果： :{color}[{verdict}]")
            st.write(explanation)
            st.caption(f"決策門檻：成本後歷史成功率 **> {success_threshold:g}%** "
                       + (f"且近期加權成功率 **> {success_threshold:g}%**；" if require_recent else "；")
                       + f"歷史樣本至少 {min_n} 筆"
                       + (f"、近期有效樣本至少 {effective_n_min:g} 筆" if require_recent else "")
                       + "。同時出現多空訊號時，採觀望。")
            if verdict in ("買入", "賣出"):
                st.warning("這是基於日線歷史頻率的研究訊號，不代表獨立驗證過的未來成功機率，也不是盤中當沖或實盤下單指令。")
            st.subheader("今天每個指標的判斷依據")
            st.dataframe(signal_table, use_container_width=True, hide_index=True)
            st.download_button("⬇️ 下載最新訊號與歷史勝率 CSV",
                               data=signal_table.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"youxixi_{code}_{signal_day}_latest_signal.csv",
                               mime="text/csv")
            st.info("此處的『賣出』表示歷史上看跌方向比較常成功，**不代表已持股應立即平倉，亦不等於可合法放空**。"
                    "每筆日線訊號在收盤後才能確認，進場假設為下一交易日開盤。"
                    "此版本未接即時 1／5 分鐘報價，不能解讀為當下可成交價格。")

with tab_results:
    if not up_to_date:
        st.info("請到左側選好股票、回測時間與指標，按「開始回測」。公式頁籤不必等資料下載即可查看。")
    else:
        prices, values, signals, results = saved["prices"], saved["values"], saved["signals"], saved["result"]
        st.success(f"{name}（{code}）｜行情代號 {saved['symbol']}｜最後可用交易日 {prices.index[-1].date()}｜已載入 {len(prices)} 根日 K")
        st.caption("每筆訊號在 t 日收盤後才成立：使用 t+1 交易日開盤價進場，t+h 交易日收盤價評估。"
                   "看跌只計算反向價格變化，不代表實際可放空；不是當沖成交紀錄。")
        st.subheader("各指標 × 方向 × 預測期限：歷史表現")
        if results.empty:
            st.warning("沒有可評估的訊號；可以延長回測期間。")
        else:
            low_n = results[(results["訊號數"] > 0) & (results["訊號數"] < min_n)]
            if len(low_n):
                st.warning(f"有 {len(low_n)} 組樣本數不足 {min_n} 次，不宜把其命中率視為可靠預測。")
            st.caption("近期加權命中率使用已完成訊號的**簡化成本後**標籤；預設成本 0%，不等於已校準的未來成功機率。")
            full = results.copy()
            full.insert(0, "股票名稱", name)
            full.insert(0, "股票代號", code)
            ordered = full.sort_values(["預測交易日", "歷史方向命中率%", "訊號數"],
                                       ascending=[True, False, False], na_position="last")
            st.dataframe(ordered, use_container_width=True, hide_index=True)
            st.download_button("⬇️ 下載回測結果 CSV（包含股票中文名稱）",
                               data=ordered.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"youxixi_{code}_{prices.index[-1].date()}_backtest.csv",
                               mime="text/csv")
        st.subheader("各指標：最後一根日 K 的訊號狀態")
        signals_now = pd.DataFrame([{
            "股票代號": code, "股票名稱": name, "技術指標": k,
            "最新收盤訊號": {1: "看漲", -1: "看跌", 0: "無新訊號"}[int(v.iloc[-1])],
            "最新交易日": str(prices.index[-1].date()),
        } for k, v in signals.items()])
        st.dataframe(signals_now, use_container_width=True, hide_index=True)
        st.subheader("股價與技術指標走勢")
        plot_set = ["Close"]
        plot_prices = prices[["Close"]].rename(columns={"Close": "收盤"}).join(
            values[["MA短", "MA長", "EMA短", "EMA長", "布林上", "布林下"]])
        st.line_chart(plot_prices.loc[plot_prices.index >= (prices.index.max() - pd.Timedelta(days=200))], height=330)
        mapping = {
            NAMES[0]: ["收盤", "MA短", "MA長"],
            NAMES[1]: ["收盤", "EMA短", "EMA長"],
            NAMES[2]: ["K", "D"],
            NAMES[3]: ["DIF", "DEA", "MACD柱"],
            NAMES[4]: ["收盤", "布林中", "布林上", "布林下"],
            NAMES[5]: ["RSI"],
            NAMES[6]: ["收盤", "Fib38.2", "Fib50", "Fib61.8"],
            NAMES[7]: ["收盤", "轉換線", "基準線", "雲A(當日可見)", "雲B(當日可見)"],
            NAMES[8]: ["Z分數"],
            NAMES[9]: ["+DI", "-DI", "ADX"],
        }
        plot_indicator = st.selectbox("個別指標曲線", list(signals.keys()))
        show = values[mapping[plot_indicator]].loc[values.index >= (values.index.max() - pd.Timedelta(days=200))]
        st.line_chart(show, height=280)
        st.info("線圖僅供檢視：斐波那契是滾動視窗規則，一目雲顯示當日已知的雲層；"
                "任何指標可能出現零訊號，並不代表程式失效。")

with tab_formulas:
    st.subheader("十種技術指標｜公式、初始值、訊號規則")
    st.markdown("你可以直接核對公式與參數。**指標數值與買賣訊號是兩件事**："
                "本版明確把每個指標轉換成可回測的交叉／突破事件，這是研究假設，不是指標本身的唯一用法。")
    st.caption("C=收盤、H=最高、L=最低；n=期數；t=本根日 K；t−1=前一根日 K。所有公式僅使用當時已知資訊。")
    for index, (title, label, formula, url) in enumerate(formula_sections(params), start=1):
        with st.expander(f"{index}. {title}｜{label}", expanded=False):
            st.markdown(formula)
            st.markdown(f"[參考資料（開啟外部網站）]({url})")
    st.divider()
    st.subheader("最新買入／賣出／觀望：判斷公式")
    st.markdown(r"""
**1. 歷史成本後成功率：**
$$ P_{hist}=\frac{\text{成本後方向報酬}>0\text{ 的已完成訊號次數}}{\text{已完成訊號總數}}\times100\% $$

**2. 近期加權成功率：**
$$ w_i=2^{-a_i/h},\quad P_{recent}=\frac{\sum_i w_i I(r_{net,i}>0)}{\sum_i w_i}\times100\% $$
$$ N_{eff}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2} $$
- $a_i$：訊號結果完成後至目前已經過的交易日；$h$：近期半衰期，預設 20 個交易日。
- $I(r_{net,i}>0)$：單筆簡化成本後方向報酬大於 0 記為 1，否則 0。

**3. 顯示判斷：** 當日有新交叉或突破，歷史完成樣本數達最低值，且 $P_{hist}$ **嚴格大於**使用者設定門檻，才可能觸發；若勾選近期確認，$P_{recent}$ 也必須大於相同門檻，且 $N_{eff}$ 達最低值。看漲合格 → 買入；看跌合格 → 賣出；多空同時合格或都未合格 → 觀望。

**統計警告：** 這是樣本內已實現命中頻率，受選股、指標選擇、重疊持有區間、資料偏差與市場變化影響；不是經獨立驗證或校準的未來成功機率。
""")
    st.divider()
    st.subheader("程式實際公式也可以看")
    st.caption("下方是本專案 GitHub 同一份程式碼的函式原文。你可將上述數學定義與程式逐條比對。")
    with st.expander("顯示指標計算函式：compute_indicators"):
        st.code(inspect.getsource(compute_indicators), language="python")
    with st.expander("顯示買賣訊號函式：make_signals"):
        st.code(inspect.getsource(make_signals), language="python")
    with st.expander("顯示 Wilder RSI / ADX 初值及遞推函式"):
        from indicators import _rsi, _adx, _wilder_seeded
        st.code(inspect.getsource(_wilder_seeded) + "\n" + inspect.getsource(_rsi)
                + "\n" + inspect.getsource(_adx), language="python")

with tab_verify:
    st.subheader("逐日核算：原始 OHLC 與十種指標的中間值")
    if not up_to_date:
        st.info("按「開始回測」後，可以在這裡檢查每一天的價格與中間計算數值，並匯出 CSV 比對 Excel 或其他券商軟體。")
    else:
        prices, values, signals = saved["prices"], saved["values"], saved["signals"]
        combined = prices.join(values).copy()
        for signal_name, series in signals.items():
            combined[f"訊號｜{signal_name}"] = series
        combined.index.name = "交易日"
        st.caption("全部資料均截至本次選定的歷史基準日。NaN 表示暖機期間數據不足、平滑未完成或數學上無定義。")
        last_n = st.slider("畫面顯示最後幾個交易日", min_value=5, max_value=100,
                           value=20, step=5)
        st.dataframe(combined.tail(last_n).T, use_container_width=True)
        st.download_button("⬇️ 下載逐日 OHLC + 全部中間計算值 CSV",
                           data=combined.to_csv(index=True).encode("utf-8-sig"),
                           file_name=f"youxixi_{code}_{prices.index[-1].date()}_indicator_details.csv",
                           mime="text/csv")
        st.markdown("**核對方式：** 例如在 Excel 先用近 20 根收盤價算 `AVERAGE`，"
                    "再用相同 20 筆算 `STDEV.P`；應對應 `布林中` 和 `布林σ`。"
                    "KD、EMA、RSI、ADX 還需遵循本版明示的起始值和遞迴方式。")

st.divider()
with st.expander("⚠️ 回測邊界、風險與下一階段"):
    st.markdown("""
- 買入／賣出／觀望標籤來自當日收盤後的新事件、歷史成本後勝率與近期加權門檻；勝率並非未來事件已經校準的機率。
- 本版十種指標均已加入**日線計算與規則化方向回測**，不是盤中 1 分／5 分 K 線當沖。
- 方向命中率不等同獲利率；更不是保證的未來機率。不同訊號可同日出現、同時測多個預測期會有**樣本重疊**。
- 回測訊號為當日收盤後才確認，假設下一交易日開盤可成交。看跌方向只以反向報酬做代理，**未模擬真實放空制度**。
- 成本模型是對方向報酬做統一扣減；尚未完整處理交易稅、手續費、滑價、點差、漲跌停、流動性與企業行動的歷史 point-in-time 版本。
- 日線資料來自 Yahoo Finance 非正式來源且可能修改；歷史資料採 `auto_adjust=True`，需留意除權息或事後調整。
- 斐波那契高低點完全使用「當時已知的滾動區間」，不是事後主觀選的波段；一目均衡表不拿落後線製造前視訊號。
- 沒有券商交易連線、沒有選擇權買賣、沒有自動下單；下一階段優先做盤中資料、可執行交易成本、外樣本驗證和訊號校準。
""")

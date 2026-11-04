"""台股短期技術指標敏感度分析：輕量 Streamlit 起始版。

這是一個研究用原型：只下載所選股票、截至所選基準日的資料；
對歷史已完成的 1/3/5/10/20 交易日方向訊號估計成功比例。
非投資建議或可直接執行的交易策略。
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import math
import re

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from stock_names import fetch_stock_names, find_stock_name

HORIZONS = [1, 3, 5, 10, 20]

st.set_page_config(page_title="油洗洗｜台股技術訊號回測", page_icon="📈", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(stock_id: str, market: str, start_date: str, end_date_exclusive: str):
    """僅即時擷取所需股票日線；end 是不包含的上界。"""
    if "." in stock_id:
        candidates = [stock_id]
    elif market == "上櫃":
        candidates = [f"{stock_id}.TWO", f"{stock_id}.TW"]
    else:
        candidates = [f"{stock_id}.TW", f"{stock_id}.TWO"]
    errors = []
    for symbol in candidates:
        try:
            df = yf.Ticker(symbol).history(
                start=start_date,
                end=end_date_exclusive,
                interval="1d",
                auto_adjust=True,
                actions=False,
                timeout=20,
            )
            if df is None or df.empty:
                continue
            if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            ix = pd.to_datetime(df.index)
            if ix.tz is not None:
                ix = ix.tz_convert("Asia/Taipei").tz_localize(None)
            df.index = ix.normalize()
            df = df[~df.index.duplicated(keep="last")].sort_index().dropna(
                subset=["Open", "High", "Low", "Close"]
            )
            if len(df) >= 35:
                return df, symbol
        except Exception as ex:
            errors.append(f"{symbol}: {str(ex)[:120]}")
    raise ValueError("無法取得足夠的歷史行情。請檢查股票代號／日期或稍後再試。" +
                     (" API 訊息：" + "; ".join(errors) if errors else ""))


def make_signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    """每日收盤後才確定的訊號；1 表示看漲，-1 表示看跌。"""
    close, high, low = df["Close"], df["High"], df["Low"]

    # RSI(14) Wilder 平滑；50 線穿越可在短區間提供相對足夠樣本。
    change = close.diff()
    average_gain = change.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    average_loss = (-change.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50)

    # KD(9,3,3)：簡單移動平均平滑的 K/D；不是所有券商的遞迴式 KD。
    ll = low.rolling(9, min_periods=9).min()
    hh = high.rolling(9, min_periods=9).max()
    rsv = (100 * (close - ll) / (hh - ll).replace(0, np.nan)).clip(0, 100)
    k = rsv.rolling(3, min_periods=3).mean()
    d = k.rolling(3, min_periods=3).mean()

    # MACD(12,26,9)：MACD 線穿越訊號線。
    fast = close.ewm(span=12, min_periods=12, adjust=False).mean()
    slow = close.ewm(span=26, min_periods=26, adjust=False).mean()
    macd = fast - slow
    macd_signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()

    def crossings(a: pd.Series, b: pd.Series) -> pd.Series:
        valid = a.notna() & b.notna() & a.shift(1).notna() & b.shift(1).notna()
        buy = valid & (a > b) & (a.shift(1) <= b.shift(1))
        sell = valid & (a < b) & (a.shift(1) >= b.shift(1))
        return pd.Series(np.select([buy, sell], [1, -1], default=0), index=a.index, dtype=int)

    level_50 = pd.Series(50.0, index=df.index)
    return {
        "RSI(14) 穿越 50": crossings(rsi, level_50),
        "KD(9,3,3) K/D 交叉": crossings(k, d),
        "MACD(12,26,9) 交叉": crossings(macd, macd_signal),
    }


def wilson_interval(wins: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval，僅供小樣本不確定性參考。"""
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.96
    p = wins / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return max(0, center - radius), min(1, center + radius)


def evaluate(df: pd.DataFrame, signals: dict[str, pd.Series], start: pd.Timestamp, horizons: list[int], half_life_days: int = 20):
    """不使用基準日後的資料：訊號 t 收盤後形成、t+1 開盤評估進場、t+h 收盤評估結局。"""
    opens = df["Open"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    indices = np.arange(len(df), dtype=int)
    within = np.asarray(df.index >= start)
    rows = []
    for horizon in horizons:
        # 最後 horizon 天的訊號沒有完成標籤，因此不能加入歷史命中率。
        eligible = indices[within & (indices + horizon < len(df))]
        if not len(eligible):
            continue
        raw_returns = closes[eligible + horizon] / opens[eligible + 1] - 1
        for name, signal in signals.items():
            values = signal.to_numpy(dtype=int)
            for direction, label in [(1, "看漲"), (-1, "看跌")]:
                positions = np.flatnonzero(values[eligible] == direction)
                n = int(len(positions))
                if n == 0:
                    continue
                direction_returns = raw_returns[positions] * direction
                wins = int(np.count_nonzero(direction_returns > 0))
                lo, hi = wilson_interval(wins, n)
                base = float(np.mean((raw_returns * direction) > 0) * 100)
                # 近期加權：權重每 half_life_days 個交易日減半。
                # 注意只使用已成熟的歷史標籤，並非真實交易記錄。
                win_flags = (direction_returns > 0).astype(float)
                signal_indices = eligible[positions]
                age_trading_days = (len(df) - 1) - signal_indices
                weights = np.power(0.5, age_trading_days / half_life_days)
                weighted_success = float(100 * np.sum(weights * win_flags) / weights.sum())
                effective_n = float(weights.sum() ** 2 / np.dot(weights, weights))
                rows.append({
                    "指標與規則": name,
                    "方向": label,
                    "預測交易日": horizon,
                    "歷史成功率%": round(100 * wins / n, 1),
                    "近期加權成功率%": round(weighted_success, 1),
                    "近期有效樣本數": round(effective_n, 1),
                    "成功／訊號": f"{wins}/{n}",
                    "訊號數": n,
                    "95%區間%": f"{100*lo:.0f}–{100*hi:.0f}",
                    "同期基準成功率%": round(base, 1),
                    "平均方向報酬%": round(100 * float(np.mean(direction_returns)), 2),
                })
    return pd.DataFrame(rows)


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def cached_stock_names():
    return fetch_stock_names()


st.title("📈 油洗洗｜台股技術訊號回測")
st.caption("v0.4｜GitHub × Streamlit 雲端版；日線方向訊號研究，當沖／選擇權尚未納入")

with st.sidebar:
    st.header("研究條件")
    stocks, stock_service_errors = cached_stock_names()
    stock_codes = sorted(stocks)
    initial_index = stock_codes.index("2454") if "2454" in stocks else 0
    selected_code = st.selectbox(
        "股票（輸入代號或中文名稱搜尋）",
        options=stock_codes,
        index=initial_index,
        format_func=lambda c: f"{stocks[c][0]}（{c}｜{stocks[c][1]}）",
        help="可在下拉選單內輸入股票代號或中文名稱進行搜尋；名稱來源為證交所與櫃買中心公開資料。",
    )
    with st.expander("查不到股票？手動輸入代號"):
        manual_code = st.text_input("自訂代號（選填）", placeholder="例如 6488 或 6488.TWO")
    stock_input = manual_code.strip().upper() or selected_code
    stock_name, stock_market = find_stock_name(stock_input, stocks)
    st.caption(f"目前標的：**{stock_name}（{stock_input}）**｜{stock_market}")
    if stock_service_errors:
        st.caption("部分官方名稱服務未連線，選單可能不完整；可手動輸入代號。")
    today_tw = datetime.now(ZoneInfo("Asia/Taipei")).date()
    asof_date = st.date_input("歷史分析基準日", value=today_tw, max_value=today_tw)
    months = st.selectbox("往前回看", options=[1, 3, 6, 12], index=1, format_func=lambda v: f"{v} 個月")
    horizons = st.multiselect("預測幾個交易日", options=HORIZONS, default=HORIZONS)
    min_samples = st.number_input("最低樣本數提醒門檻", min_value=1, max_value=100, value=20, step=1)
    half_life_days = st.number_input("近期表現加權半衰期（交易日）", min_value=1, max_value=100, value=20,
                                     help="20 表示距今 20 個交易日前的已完成訊號，權重約為今日的一半。")
    analyze = st.button("開始回測", type="primary", use_container_width=True)


# 下載 CSV 或展開說明會讓 Streamlit 重新執行；以 session_state 保留結果。
condition_key = (
    stock_input.strip().upper(), str(asof_date), int(months),
    tuple(sorted(horizons)), int(half_life_days),
)

if analyze:
    stock = stock_input.strip().upper()
    if not re.fullmatch(r"\d{4,6}(?:\.(?:TW|TWO))?", stock):
        st.error("請輸入台股代碼，例如 2454、2330.TW 或 6488.TWO。")
        st.stop()
    if not horizons:
        st.warning("請至少選擇一個預測交易日數。")
        st.stop()

    cutoff = pd.Timestamp(asof_date) - pd.DateOffset(months=months)
    # 先取得指標所需的暖機期；歷史基準日以後的資料不下載。
    fetch_start = cutoff - pd.Timedelta(days=180)
    fetch_end = pd.Timestamp(asof_date) + timedelta(days=1)  # yfinance 的 end 不包含該日

    with st.spinner("正在取得單一股票行情並計算歷史訊號……"):
        try:
            prices, resolved_symbol = load_prices(
                stock, stock_market, str(fetch_start.date()), str(fetch_end.date())
            )
            prices = prices.loc[prices.index <= pd.Timestamp(asof_date)].copy()
            if prices.empty or prices.index.max() < cutoff:
                raise ValueError("選定日期範圍沒有足夠資料。")
            signals = make_signals(prices)
            results = evaluate(prices, signals, cutoff, horizons, int(half_life_days))
        except Exception as ex:
            st.session_state.pop("youxixi_result", None)
            st.error(f"回測未完成：{ex}")
            st.stop()

    latest = [
        {
            "股票代號": stock, "股票名稱": stock_name, "指標": name,
            "當時收盤訊號": {1: "看漲", -1: "看跌", 0: "無新交叉"}[int(seq.iloc[-1])],
        }
        for name, seq in signals.items()
    ]
    st.session_state["youxixi_result"] = {
        "key": condition_key,
        "code": stock,
        "name": stock_name,
        "symbol": resolved_symbol,
        "last_session": str(prices.index[-1].date()),
        "price_rows": len(prices),
        "results": results,
        "latest": pd.DataFrame(latest),
    }

saved = st.session_state.get("youxixi_result")
if saved and saved.get("key") == condition_key:
    stock = saved["code"]
    stock_name = saved["name"]
    last_session = saved["last_session"]
    results = saved["results"]
    st.success(
        f"已取得 {stock_name}（{stock}；Yahoo：{saved['symbol']}）日線，"
        f"最新可用交易日：{last_session}。本次抓取 {saved['price_rows']:,} 筆日線。"
    )
    st.info(
        "規則：t 日收盤產生訊號，t+1 日開盤作為基準，t+h 日收盤評估漲跌。"
        "這是日線『方向』命中率，不是盤中當沖績效，且未扣費稅滑價。"
    )
    st.subheader(f"{stock_name}（{stock}）｜各指標 × 預測期歷史成功率")
    if results.empty:
        st.warning("目前沒有足夠且已完成驗證的訊號，可延長回測區間或調整指標規則。")
    else:
        small = results[results["訊號數"] < min_samples]
        if not small.empty:
            st.warning(
                f"有 {len(small)} 個指標／方向／期限組合的樣本數少於 {min_samples} 次。"
                "這類百分比只宜作粗略描述。"
            )
        display_results = results.copy()
        display_results.insert(0, "股票名稱", stock_name)
        display_results.insert(0, "股票代號", stock)
        st.dataframe(
            display_results.sort_values(
                ["預測交易日", "歷史成功率%"], ascending=[True, False]
            ), use_container_width=True, hide_index=True
        )
        st.download_button(
            "下載回測 CSV（含中文名稱）",
            data=display_results.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{stock}_{last_session}_backtest.csv", mime="text/csv",
        )
    st.subheader(f"{stock_name}（{stock}）｜基準日最新收盤訊號")
    st.dataframe(saved["latest"], hide_index=True, use_container_width=True)
    st.caption(
        "此表只顯示收盤已成立的訊號；不能假設可在當日收盤價成交。"
        "近期加權值不是已校準的未來勝率。"
    )
elif saved:
    st.info("研究條件已變更，請按『開始回測』更新結果。")
else:
    st.info("請先在左邊選股票、歷史日期、回看長度及預測交易日數，再按『開始回測』。")

with st.expander("回測解讀與資料限制（重要）"):
    st.markdown("""
- **本系統只找歷史基準日當下的局部有效性**，不要求該指標以後永遠有效。
- 「歷史成功率」是觀察期間內訊號的命中比例，**不是下一筆交易成功的保證或經校準的機率預測**。
- 看跌成功代表股價在評估區間下降，**不表示台股可以無限制放空，也不代表實際賣出後就獲利**。
- 目前示範規則：RSI(14) 穿越 50；KD(9,3,3) K/D 交叉；MACD(12,26,9) 交叉。不同訊號規則會產生不同結果。
- 歷史基準日之後的行情**不會用來計算**；若某訊號在基準日前尚未完成所選 h 個交易日，便不納入統計。
- 每個 horizon 下最後 h 個交易日不具備已完成的預測結果，因此 20 日欄位的樣本通常比 1 日少。
- 僅使用 Yahoo Finance 非正式行情介面，若資料源不穩定會有抓取失敗、延遲或修正的可能；股價採自動調整 OHLC。
- 尚未處理完整交易成本、除權息事件的 point-in-time 還原版本、成交限制、樣本重疊造成的相依性，正式交易前必須補強。
- 此原型不儲存行情或交易資料；Streamlit 雲端重啟後，快取及本次回測結果可能消失。
""")


st.divider()
st.header("接下來的開發方向")
st.markdown("""
1. **日線回測引擎（目前）**：股票代號＋中文名稱、RSI／KD／MACD、1／3／5／10／20 交易日、近期加權、樣本數與下載 CSV。
2. **盤中當沖資料（尚未完成）**：1 分鐘／5 分鐘線、可成交的訊號時間點、手續費、交易稅與滑價。
3. **策略驗證（尚未完成）**：分開訓練／測試區間、樣本重疊處理、實際交易紀錄、盈虧歸因、自我校準及連敗停用規則。

**選擇權暫不納入**，不影響現階段的股票訊號研究。等回測引擎成熟，再另行決定要不要開發。
""")

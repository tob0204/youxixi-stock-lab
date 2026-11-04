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
from risk import check_long_txo, long_option_pnl, monthly_account_action

HORIZONS = [1, 3, 5, 10, 20]

st.set_page_config(page_title="油洗洗｜台股量化實驗室", page_icon="📈", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(stock_id: str, start_date: str, end_date_exclusive: str):
    """僅即時擷取所需股票日線；end 是不包含的上界。"""
    candidates = ([stock_id] if "." in stock_id else [f"{stock_id}.TW", f"{stock_id}.TWO"])
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


st.title("📈 油洗洗｜台股量化實驗室")
st.caption("第一版原型｜不建股價資料庫｜API 按需取用｜只使用指定歷史基準日當時可得的收盤資料")

with st.sidebar:
    st.header("研究條件")
    stock_input = st.text_input("股票代號", value="2454", help="例如 2330、2454、3665、6139；也可填 2330.TW")
    today_tw = datetime.now(ZoneInfo("Asia/Taipei")).date()
    asof_date = st.date_input("歷史分析基準日", value=today_tw, max_value=today_tw)
    months = st.selectbox("往前回看", options=[1, 3, 6, 12], index=1, format_func=lambda v: f"{v} 個月")
    horizons = st.multiselect("預測幾個交易日", options=HORIZONS, default=HORIZONS)
    min_samples = st.number_input("最低樣本數提醒門檻", min_value=1, max_value=100, value=20, step=1)
    half_life_days = st.number_input("近期表現加權半衰期（交易日）", min_value=1, max_value=100, value=20,
                                     help="20 表示距今 20 個交易日前的已完成訊號，權重約為今日的一半。")
    analyze = st.button("開始回測", type="primary", use_container_width=True)

if analyze:
    stock = stock_input.strip().upper()
    if not re.fullmatch(r"\d{4,6}(?:\.(?:TW|TWO))?", stock):
        st.error("請輸入台股代碼，例如 2454、2330.TW 或 6488.TWO。")
        st.stop()
    if not horizons:
        st.warning("請至少選擇一個預測交易日數。")
        st.stop()

    cutoff = pd.Timestamp(asof_date) - pd.DateOffset(months=months)
    # 暖機期：技術指標需使用觀察窗之前的歷史價格，避免一開始指標尚未成形。
    fetch_start = cutoff - pd.Timedelta(days=180)
    fetch_end = pd.Timestamp(asof_date) + timedelta(days=1)  # yfinance end 不包含該日

    with st.spinner("正在取得單一股票行情並計算歷史訊號……"):
        try:
            prices, resolved_symbol = load_prices(stock, str(fetch_start.date()), str(fetch_end.date()))
            prices = prices.loc[prices.index <= pd.Timestamp(asof_date)].copy()
            if prices.empty or prices.index.max() < cutoff:
                raise ValueError("選定日期範圍沒有足夠資料。")
            signals = make_signals(prices)
            results = evaluate(prices, signals, cutoff, horizons, int(half_life_days))
        except Exception as ex:
            st.error(str(ex))
            st.stop()

    last_session = prices.index[-1].date()
    st.success(f"已取得 {resolved_symbol} 日線，最新可用交易日：{last_session}。本次僅抓取 {len(prices):,} 筆日線。");
    st.info("交易規則：t 日收盤產生訊號，t+1 日開盤作為基準，t+h 日收盤評估漲跌。這是日線『方向』命中率，不是盤中當沖績效、選擇權勝率，且未扣費稅滑價。")
    st.subheader("各指標 × 預測期的歷史成功比例（含近期加權）")
    if results.empty:
        st.warning("目前沒有足夠且已完成驗證的訊號，可延長回測區間或調整指標規則。")
    else:
        small = results[results["訊號數"] < min_samples]
        if not small.empty:
            st.warning(f"有 {len(small)} 個指標／方向／期限組合的樣本數少於 {min_samples} 次，百分比只能視為極粗略的描述。")
        st.dataframe(results.sort_values(["預測交易日", "歷史成功率%"], ascending=[True, False]), use_container_width=True, hide_index=True)
        st.download_button("下載結果 CSV", results.to_csv(index=False).encode("utf-8-sig"), file_name=f"{resolved_symbol}_{last_session}_sensitivity.csv", mime="text/csv")
    st.subheader("基準日最新收盤訊號")
    latest = [{"指標": name, "當時收盤訊號": {1: "看漲", -1: "看跌", 0: "無新交叉"}[int(seq.iloc[-1])]} for name, seq in signals.items()]
    st.dataframe(pd.DataFrame(latest), hide_index=True, use_container_width=True)
    st.caption("此表只能在最新交易日收盤後成立；出現訊號不代表已按當日收盤價成交。近期加權僅供對照，不足以直接下單。")
else:
    st.info("請先在左邊選股票、歷史日期、回看長度及未來交易日數，然後點「開始回測」。")

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
- 此原型不儲存行情資料；Streamlit 的 1 小時記憶體快取可能因服務重新啟動而消失。
""")


st.divider()
st.header("🧮 資金防火牆｜臺指選擇權買方試算")
st.warning("此區僅模擬單純買進臺指選擇權（TXO）一口或多口的權利金與現金支出，**不是當沖策略回測**。不支援選擇權賣方、賣方價差組合或其他保證金交易。")
col_a, col_b, col_c = st.columns(3)
with col_a:
    monthly_cap = st.number_input("本月交易帳戶現金上限 NT$", min_value=1000, max_value=100000, value=10000, step=1000)
    cash_now = st.number_input("目前可用交易現金 NT$", min_value=0, max_value=100000, value=10000, step=100)
with col_b:
    single_trade_cap = st.number_input("單筆投入上限 NT$（含預估費用）", min_value=100, max_value=100000, value=1500, step=100)
    premium_points = st.number_input("TXO 買進成交權利金（點）", min_value=0.1, max_value=10000.0, value=20.0, step=1.0)
with col_c:
    contracts = st.number_input("買進口數", min_value=1, max_value=100, value=1, step=1)
    fee_side = st.number_input("每口單邊手續費及稅費估計 NT$", min_value=0.0, max_value=1000.0, value=30.0, step=5.0, help="請輸入實際券商報價和稅費合計；30 僅為示範值。")

check = check_long_txo(option_premium_points=float(premium_points), contracts=int(contracts),
                       cash_available=min(float(cash_now), float(monthly_cap)),
                       single_trade_limit=float(single_trade_cap),
                       fee_per_side_per_contract=float(fee_side))
metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("權利金總額", f"NT$ {check.premium_paid:,.0f}")
metric_b.metric("預留雙邊費用後投入", f"NT$ {check.reserved_cash:,.0f}")
metric_c.metric("剩餘可用現金", f"NT$ {check.budget_remaining:,.0f}")
if check.affordable and check.within_trade_limit:
    st.success("通過『現金上限』及『單筆投入上限』兩項檢查；不代表具備正期望值，也不代表應下單。")
else:
    st.error("未通過：現金或單筆限額不足。不可透過追加保證金、借款或賣出選擇權繞過限制。")

st.subheader("買進後不同平倉權利金的損益情境")
example_exit = st.number_input("假設賣出平倉權利金（點）", min_value=0.0, max_value=10000.0, value=30.0, step=1.0)
example_pnl = long_option_pnl(entry_points=float(premium_points), exit_points=float(example_exit),
                              contracts=int(contracts), fee_per_side_per_contract=float(fee_side))
st.metric("示意已實現淨損益（預估雙邊費用後）", f"NT$ {example_pnl:+,.0f}")
st.caption("臺指選擇權（TXO）每點 NT$50。正確績效應以實際成交價、買賣價差、手續費及稅費計算；到期歸零可能損失全部權利金及已支付成本。")

withdraw, retained = monthly_account_action(float(cash_now), float(monthly_cap))
st.info(f"若現在依『超過風險本金即轉出』規則結帳，建議轉至長期投資資產池 NT$ {withdraw:,.0f}，交易帳戶最多保留 NT$ {retained:,.0f}。本模擬不會執行轉帳或下單。")

with st.expander("油洗洗現階段功能 vs 仍待開發的真實當沖功能"):
    st.markdown("""
**現階段已具備**：日線 RSI／KD／MACD 訊號、1／3／5／10／20 交易日歷史方向命中比例、近期半衰期加權、Wilson 區間、TXO 買方資金上限試算。

**還沒完成，絕不可混為一談**：逐分鐘／逐筆當沖行情、盤中形成與成交時點、真實委託簿/滑價、買賣價差、TXO 各履約價/到期日權利金歷史、IV/時間價值、選擇權實際淨報酬回測、完整的實盤日誌、自動停損／警報、走勢變動下的外樣本校準。

**實作次序**：先在模擬紀錄中驗證日線訊號→補上盤中交易資料和真實選擇權價格→完成成本後的外樣本績效與停用規則→小額實盤。不得用標的漲跌命中率替代選擇權實際賺錢的機率。
""")

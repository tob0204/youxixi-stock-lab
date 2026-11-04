"""油洗洗 v0.6：十種技術指標，全部只使用當日或更早的 OHLC 資料。

各指標的資料窗、初始值與訊號規則必須與 app.py 的「計算公式」頁籤一致。
研究用途，尚非盤中當沖策略。避免未來資訊外洩：一目雲層使用 shift(+26)；
斐波那契 swing 使用當時可得的 rolling window，不使用事後確認的轉折。
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

DEFAULTS: dict[str, int | float] = {
    "ma_fast": 5, "ma_slow": 20,
    "ema_fast": 12, "ema_slow": 26,
    "kd_window": 9, "kd_smooth": 3,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "bb_window": 20, "bb_k": 2.0,
    "rsi_window": 14,
    "fib_window": 60,
    "ichi_conversion": 9, "ichi_base": 26, "ichi_span_b": 52, "ichi_shift": 26,
    "std_window": 20, "std_z": 1.0,
    "adx_window": 14, "adx_threshold": 25.0,
}

NAMES = ["MA 移動平均線", "EMA 指數平滑移動平均線", "KD 隨機指標", "MACD 平滑異同移動平均線",
         "布林通道", "RSI 相對強弱指數", "斐波那契回調", "一目均衡表", "標準差（Z-score）", "ADX 趨勢強度"]


def _wilder_seeded(values: pd.Series, period: int, first_index: int) -> pd.Series:
    """第一筆為 first_index 之前 period 筆有效數據的 SMA，再用 Wilder RMA 遞推。"""
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    begin = first_index - period + 1
    if begin < 0 or first_index >= len(arr):
        return pd.Series(out, index=values.index)
    seed = arr[begin:first_index + 1]
    if not np.isfinite(seed).all():
        return pd.Series(out, index=values.index)
    out[first_index] = float(seed.mean())
    for i in range(first_index + 1, len(arr)):
        if np.isfinite(arr[i]):
            out[i] = (out[i - 1] * (period - 1) + arr[i]) / period
    return pd.Series(out, index=values.index, dtype=float)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = _wilder_seeded(gain, period, first_index=period)
    avg_loss = _wilder_seeded(loss, period, first_index=period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    result = result.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return result


def _kd(high: pd.Series, low: pd.Series, close: pd.Series, n: int, smooth: int):
    lo = low.rolling(n, min_periods=n).min()
    hi = high.rolling(n, min_periods=n).max()
    rsv = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    rsv = rsv.clip(0, 100).fillna(50.0)
    # 只在第一個 n 日區間形成後才提供 K/D；K、D 均以 50 為前值。
    k = np.full(len(close), np.nan)
    d = np.full(len(close), np.nan)
    for i in range(n - 1, len(close)):
        kp = 50.0 if i == n - 1 else k[i - 1]
        dp = 50.0 if i == n - 1 else d[i - 1]
        k[i] = ((smooth - 1) * kp + rsv.iloc[i]) / smooth
        d[i] = ((smooth - 1) * dp + k[i]) / smooth
    return rsv.where(pd.Series(range(len(close)), index=close.index) >= n - 1), pd.Series(k, index=close.index), pd.Series(d, index=close.index)


def _fib(high: pd.Series, low: pd.Series, n: int) -> pd.DataFrame:
    """滾動區間的最後一個最高點／最低點定義趨勢；不依賴事後 swing 高低點。"""
    hi = high.rolling(n, min_periods=n).max()
    lo = low.rolling(n, min_periods=n).min()
    direction = np.zeros(len(high), dtype=int)
    for t in range(n - 1, len(high)):
        wh = high.iloc[t - n + 1:t + 1].to_numpy()
        wl = low.iloc[t - n + 1:t + 1].to_numpy()
        if not (np.isfinite(wh).all() and np.isfinite(wl).all()) or hi.iloc[t] <= lo.iloc[t]:
            continue
        hi_position = np.flatnonzero(wh == np.max(wh))[-1]
        lo_position = np.flatnonzero(wl == np.min(wl))[-1]
        direction[t] = 1 if lo_position < hi_position else -1 if hi_position < lo_position else 0
    trend = pd.Series(direction, index=high.index)
    spread = hi - lo
    def level(ratio):
        return pd.Series(np.where(trend == 1, hi - ratio * spread,
                                  np.where(trend == -1, lo + ratio * spread, np.nan)),
                         index=high.index, dtype=float)
    return pd.DataFrame({"Fib最高": hi, "Fib最低": lo, "Fib趨勢": trend,
                         "Fib23.6": level(0.236), "Fib38.2": level(0.382),
                         "Fib50": level(0.5), "Fib61.8": level(0.618), "Fib78.6": level(0.786)})


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int):
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=close.index, dtype=float)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=close.index, dtype=float)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    tr.iloc[0] = plus_dm.iloc[0] = minus_dm.iloc[0] = np.nan
    smooth_tr = _wilder_seeded(tr, n, first_index=n)
    smooth_plus = _wilder_seeded(plus_dm, n, first_index=n)
    smooth_minus = _wilder_seeded(minus_dm, n, first_index=n)
    plus_di = 100 * smooth_plus / smooth_tr.replace(0, np.nan)
    minus_di = 100 * smooth_minus / smooth_tr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    # 首個 ADX = 從第 n 日至第 (2n-1) 日的 n 個 DX 平均
    adx = _wilder_seeded(dx, n, first_index=2 * n - 1)
    return tr, plus_di, minus_di, dx, adx


def compute_indicators(prices: pd.DataFrame, params: Mapping[str, int | float] | None = None) -> pd.DataFrame:
    p = {**DEFAULTS, **(dict(params) if params else {})}
    close = prices["Close"].astype(float)
    high = prices["High"].astype(float)
    low = prices["Low"].astype(float)
    v = pd.DataFrame(index=prices.index)
    v["收盤"] = close
    for stem in ("ma", "ema"):
        for speed in ("fast", "slow"):
            n = int(p[f"{stem}_{speed}"])
            col = f"{stem.upper()}{'短' if speed == 'fast' else '長'}"
            v[col] = (close.rolling(n, min_periods=n).mean() if stem == "ma" else
                      close.ewm(span=n, min_periods=n, adjust=False).mean())
    v["RSV"], v["K"], v["D"] = _kd(high, low, close, int(p["kd_window"]), int(p["kd_smooth"]))
    fast = close.ewm(span=int(p["macd_fast"]), min_periods=int(p["macd_fast"]), adjust=False).mean()
    slow = close.ewm(span=int(p["macd_slow"]), min_periods=int(p["macd_slow"]), adjust=False).mean()
    v["DIF"] = fast - slow
    v["DEA"] = v["DIF"].ewm(span=int(p["macd_signal"]), min_periods=int(p["macd_signal"]), adjust=False).mean()
    v["MACD柱"] = v["DIF"] - v["DEA"]
    n = int(p["bb_window"])
    v["布林中"] = close.rolling(n, min_periods=n).mean()
    v["布林σ"] = close.rolling(n, min_periods=n).std(ddof=0)
    v["布林上"] = v["布林中"] + float(p["bb_k"]) * v["布林σ"]
    v["布林下"] = v["布林中"] - float(p["bb_k"]) * v["布林σ"]
    v["RSI"] = _rsi(close, int(p["rsi_window"]))
    v = pd.concat([v, _fib(high, low, int(p["fib_window"]))], axis=1)
    conv = int(p["ichi_conversion"])
    base = int(p["ichi_base"])
    span_b = int(p["ichi_span_b"])
    shift = int(p["ichi_shift"])
    v["轉換線"] = (high.rolling(conv, min_periods=conv).max() + low.rolling(conv, min_periods=conv).min()) / 2
    v["基準線"] = (high.rolling(base, min_periods=base).max() + low.rolling(base, min_periods=base).min()) / 2
    unshifted_a = (v["轉換線"] + v["基準線"]) / 2
    unshifted_b = (high.rolling(span_b, min_periods=span_b).max() + low.rolling(span_b, min_periods=span_b).min()) / 2
    # 重要：雲是「先行繪製」未來 26 根，回測第 t 根看到的是 t-shift 的計算結果。
    v["雲A(當日可見)"] = unshifted_a.shift(shift)
    v["雲B(當日可見)"] = unshifted_b.shift(shift)
    n = int(p["std_window"])
    v["標準差均值"] = close.rolling(n, min_periods=n).mean()
    v["標準差σ"] = close.rolling(n, min_periods=n).std(ddof=0)
    v["Z分數"] = (close - v["標準差均值"]) / v["標準差σ"].replace(0, np.nan)
    v["TR"], v["+DI"], v["-DI"], v["DX"], v["ADX"] = _adx(high, low, close, int(p["adx_window"]))
    return v


def _cross(a: pd.Series, b: pd.Series) -> pd.Series:
    valid = a.notna() & b.notna() & a.shift().notna() & b.shift().notna()
    up = valid & a.gt(b) & a.shift().le(b.shift())
    down = valid & a.lt(b) & a.shift().ge(b.shift())
    return pd.Series(np.select([up, down], [1, -1], default=0), index=a.index, dtype=int)


def make_signals(values: pd.DataFrame, params: Mapping[str, int | float] | None = None) -> dict[str, pd.Series]:
    p = {**DEFAULTS, **(dict(params) if params else {})}
    close = values["收盤"]
    rsi50 = pd.Series(50.0, index=values.index)
    bbuy = (close > values["布林上"]) & (close.shift() <= values["布林上"].shift())
    bsell = (close < values["布林下"]) & (close.shift() >= values["布林下"].shift())
    fibbuy = ((values["Fib趨勢"] == 1) & (values["Fib趨勢"].shift() == 1)
              & close.gt(values["Fib61.8"]) & close.shift().le(values["Fib61.8"].shift()))
    fibsell = ((values["Fib趨勢"] == -1) & (values["Fib趨勢"].shift() == -1)
               & close.lt(values["Fib61.8"]) & close.shift().ge(values["Fib61.8"].shift()))
    top = values[["雲A(當日可見)", "雲B(當日可見)"]].max(axis=1)
    bottom = values[["雲A(當日可見)", "雲B(當日可見)"]].min(axis=1)
    cloud_ok = values[["雲A(當日可見)", "雲B(當日可見)"]].notna().all(axis=1)
    cloudbuy = cloud_ok & cloud_ok.shift(fill_value=False) & close.gt(top) & close.shift().le(top.shift())
    cloudsell = cloud_ok & cloud_ok.shift(fill_value=False) & close.lt(bottom) & close.shift().ge(bottom.shift())
    z = values["Z分數"]
    zlimit = float(p["std_z"])
    zbuy = z.gt(-zlimit) & z.shift().le(-zlimit)
    zsell = z.lt(zlimit) & z.shift().ge(zlimit)
    di_cross = _cross(values["+DI"], values["-DI"])
    adx_strength = values["ADX"] >= float(p["adx_threshold"])
    di_cross = di_cross.where(adx_strength, 0).astype(int)
    def flags(buy, sell):
        return pd.Series(np.select([buy & ~sell, sell & ~buy], [1, -1], default=0), index=close.index, dtype=int)
    return {
        NAMES[0]: _cross(values["MA短"], values["MA長"]),
        NAMES[1]: _cross(values["EMA短"], values["EMA長"]),
        NAMES[2]: _cross(values["K"], values["D"]),
        NAMES[3]: _cross(values["DIF"], values["DEA"]),
        NAMES[4]: flags(bbuy, bsell),
        NAMES[5]: _cross(values["RSI"], rsi50),
        NAMES[6]: flags(fibbuy, fibsell),
        NAMES[7]: flags(cloudbuy, cloudsell),
        NAMES[8]: flags(zbuy, zsell),
        NAMES[9]: di_cross,
    }


def validate_params(p: Mapping[str, int | float]):
    for pair in (("ma_fast", "ma_slow"), ("ema_fast", "ema_slow"), ("macd_fast", "macd_slow")):
        if p[pair[0]] >= p[pair[1]]:
            raise ValueError(f"{pair[0]} 必須小於 {pair[1]}。")
    if p["ichi_conversion"] > p["ichi_base"] or p["ichi_base"] > p["ichi_span_b"]:
        raise ValueError("一目均衡表需要轉換線期間 ≤ 基準線期間 ≤ 先行 B 期間。")
    if any(float(value) <= 0 for value in p.values()):
        raise ValueError("指標參數必須大於 0。")

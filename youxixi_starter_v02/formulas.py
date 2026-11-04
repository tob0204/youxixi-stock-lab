"""網頁公式與明確的訊號定義。與 indicators.py v0.6 維持同步。"""
from __future__ import annotations

from typing import Mapping

from indicators import NAMES

# 這裡保存公式的核對文檔；每個指標都有：原始定義、精確版本、訊號規則、注意事項。
# \\ in raw strings is handled by Python r-string below to preserve Streamlit LaTex.

def formula_sections(params: Mapping[str, int | float]):
    p = params
    data = [
        (NAMES[0], f"SMA({p['ma_fast']}) 與 SMA({p['ma_slow']})", r"""
$$ SMA_n(t)=\frac{1}{n}\sum_{i=0}^{n-1} C_{t-i} $$

- **使用價格：** 收盤價 $C$；需要完整的 $n$ 根 K 線，否則為空值。
- **訊號：** 短 MA 從下向上穿越長 MA → 看漲；反向 → 看跌。比較當日與前一日的兩條均線。
- **不是**「只要短均線位在長均線上方就天天買」。只有發生交叉才記一次訊號。
""", "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential"),
        (NAMES[1], f"EMA({p['ema_fast']}) 與 EMA({p['ema_slow']})", r"""
$$ EMA_t=\alpha C_t+(1-\alpha)EMA_{t-1},\quad \alpha=\frac{2}{n+1} $$

- **初值：** 本版使用「資料起點的第一個收盤價」當 $EMA_0$，並等到滿 $n$ 根 K 線才顯示；程式實作 `ewm(span=n, adjust=False, min_periods=n)`。
- **訊號：** 短 EMA 上穿長 EMA → 看漲；下穿 → 看跌。
- **差異提醒：** 部分券商把最初 $n$ 根收盤的 SMA 當 EMA 初值；兩者開頭數值會略有不同，增加暖機期可降低差距。
""", "https://pandas.pydata.org/docs/user_guide/window.html"),
        (NAMES[2], f"臺灣常見的遞迴 KD({p['kd_window']},{p['kd_smooth']},{p['kd_smooth']})", r"""
$$ RSV_t=100\times\frac{C_t-L_n(t)}{H_n(t)-L_n(t)} $$
$$ K_t=\frac{m-1}{m}K_{t-1}+\frac{1}{m}RSV_t $$
$$ D_t=\frac{m-1}{m}D_{t-1}+\frac{1}{m}K_t $$

- $H_n$ / $L_n$ 是最近 $n$ 根 K 線最高價／最低價；$m$ 是平滑期數。
- **初值：** 第一個有效 RSV 前的 $K_{-1}=D_{-1}=50$；若區間最高價等於最低價，該日 RSV 定義為 50。
- **訊號：** K 上穿 D → 看漲；K 下穿 D → 看跌。
- **非常重要：** 這是臺灣常見的 **50 起始遞迴 KD**，不是 StockCharts 的 slow stochastic（以 SMA 平滑 K/D）。兩者結果可能不同；不能混用。
""", "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stochastic-oscillator-fast-slow-and-full"),
        (NAMES[3], f"MACD({p['macd_fast']},{p['macd_slow']},{p['macd_signal']})", r"""
$$ DIF_t=EMA_{fast}(C)_t-EMA_{slow}(C)_t $$
$$ DEA_t=EMA_{signal}(DIF)_t $$
$$ Histogram_t=DIF_t-DEA_t $$

- **EMA 初始化：** 同上，`adjust=False` 且先滿足最少期數；DEA 只對已存在的 DIF 值做 EMA。
- **訊號：** DIF 上穿 DEA → 看漲；DIF 下穿 DEA → 看跌。
- **券商差異：** 有些系統的 MACD 柱為 $2(DIF-DEA)$，本版為 **1 倍差值**；柱的數值不同但交叉方向不變。
""", "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator"),
        (NAMES[4], f"布林通道({p['bb_window']},{p['bb_k']})", r"""
$$ Middle_t=SMA_n(C)_t $$
$$ \sigma_t=\sqrt{\frac{1}{n}\sum_{i=0}^{n-1}(C_{t-i}-Middle_t)^2} $$
$$ Upper_t=Middle_t+k\sigma_t,\quad Lower_t=Middle_t-k\sigma_t $$

- **標準差定義：** 本版使用 **母體標準差** (`ddof=0`)，不是樣本標準差 `ddof=1`。
- **訊號：** 收盤價首次突破上軌（前一天在上軌或以下）→ 看漲；首次跌破下軌 → 看跌。
- **注意：** 此為趨勢突破型規則，不是碰上軌就放空的均值回歸規則；兩種策略會給相反訊號。
""", "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/bollinger-bands"),
        (NAMES[5], f"Wilder RSI({p['rsi_window']})", r"""
$$ Gain_t=\max(C_t-C_{t-1},0),\quad Loss_t=\max(C_{t-1}-C_t,0) $$
$$ AvgGain_n=\frac{\sum_{i=1}^n Gain_i}{n},\quad AvgLoss_n=\frac{\sum_{i=1}^n Loss_i}{n} $$
$$ AvgGain_t=\frac{(n-1)AvgGain_{t-1}+Gain_t}{n} $$
$$ AvgLoss_t=\frac{(n-1)AvgLoss_{t-1}+Loss_t}{n} $$
$$ RS_t=\frac{AvgGain_t}{AvgLoss_t},\quad RSI_t=100-\frac{100}{1+RS_t} $$

- **初值：** 首 $n$ 筆價格變化之平均，往後做 Wilder 1/n 平滑。零虧損且有漲幅 → RSI=100；零漲幅且有跌幅 → RSI=0；皆為零 → RSI=50。
- **訊號：** RSI 上穿 50 → 看漲；下穿 50 → 看跌。不是 RSI<30 一律買入。
""", "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi"),
        (NAMES[6], f"過去 {p['fib_window']} 根 K 線的滾動斐波那契回調", r"""
$$ H_t=\max(High_{t-n+1:t}),\quad L_t=\min(Low_{t-n+1:t}) $$
$$ Retracement_{up}(r)=H_t-r(H_t-L_t) $$
$$ Retracement_{down}(r)=L_t+r(H_t-L_t) $$

- **回調比例：** 23.6%、38.2%、50%、61.8%、78.6%；其中 **50% 是交易慣用比例，非數學上的費波那契比率**。
- **趨勢定義（本版客製化）：** 比較最近 $n$ 根內最後一次最高價與最後一次最低價發生的先後；低點較早→上行、高點較早→下行。
- **訊號：** 上行趨勢時收盤價從下向上穿越當時的 61.8% 回調線 → 看漲；下行趨勢時從上向下穿越 61.8% 線 → 看跌。需前後兩天皆同趨勢。
- **限制：** 這是**滾動視窗的可回測代理規則**，不是人手畫的主觀波段；高低點只取當時已知資料，絕不事後挑選未來高低點。
""", "https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-annotation-tools/fibonacci-retracements"),
        (NAMES[7], f"一目均衡表({p['ichi_conversion']},{p['ichi_base']},{p['ichi_span_b']})，位移 {p['ichi_shift']} 根", r"""
$$ Tenkan_t=\frac{\max(High,t\!\!:\!t-n_1+1)+\min(Low,t\!\!:\!t-n_1+1)}{2} $$
$$ Kijun_t=\frac{\max(High,t\!\!:\!t-n_2+1)+\min(Low,t\!\!:\!t-n_2+1)}{2} $$
$$ SpanA^{raw}_t=(Tenkan_t+Kijun_t)/2 $$
$$ SpanB^{raw}_t=\frac{HH_{n_3}(t)+LL_{n_3}(t)}{2} $$
$$ CloudA_{visible,t}=SpanA^{raw}_{t-s},\quad CloudB_{visible,t}=SpanB^{raw}_{t-s} $$

- $n_1,n_2,n_3$ 分別為 9、26、52；$s$ 預設 26。**先行雲在圖上往未來畫 26 根，但今天可見的雲是 26 根以前算出的值**。
- **落後線：** 今日收盤價繪在 26 根以前的位置，**只供視覺展示，本版不拿來生成歷史訊號**（避免把後來才知道的收盤價當作過去已知）。
- **訊號：** 收盤價上穿今天可見的雲頂 → 看漲；跌破今天可見的雲底 → 看跌。
""", "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/ichimoku-cloud"),
        (NAMES[8], f"{p['std_window']} 根收盤價母體標準差與 Z 分數", r"""
$$ \mu_t=\frac{1}{n}\sum_{i=0}^{n-1}C_{t-i} $$
$$ \sigma_t=\sqrt{\frac{1}{n}\sum_{i=0}^{n-1}(C_{t-i}-\mu_t)^2} $$
$$ Z_t=\frac{C_t-\mu_t}{\sigma_t} $$

- **母體標準差：** `rolling(n).std(ddof=0)`；零波動期間 Z 分數不定義，留空。
- **訊號（本版選定的均值回歸規則）：** Z 自下向上穿越 $-z$ → 看漲；Z 自上向下穿越 $+z$ → 看跌。預設 $z=1$。
- **重要：** 標準差**只有波動幅度，沒有漲跌方向**；加入 Z 分數及穿越條件後才形成可測的方向規則。
""", "https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.std.html"),
        (NAMES[9], f"Wilder ADX({p['adx_window']})，強度門檻 {p['adx_threshold']}", r"""
$$ Up=High_t-High_{t-1},\quad Down=Low_{t-1}-Low_t $$
$$ +DM=\begin{cases}Up&Up>Down\land Up>0\\0&otherwise\end{cases} $$
$$ -DM=\begin{cases}Down&Down>Up\land Down>0\\0&otherwise\end{cases} $$
$$ TR_t=\max(High_t-Low_t,|High_t-C_{t-1}|,|Low_t-C_{t-1}|) $$
$$ +DI_t=100\frac{RMA_n(+DM)_t}{RMA_n(TR)_t},\quad -DI_t=100\frac{RMA_n(-DM)_t}{RMA_n(TR)_t} $$
$$ DX_t=100\frac{|+DI_t-(-DI_t)|}{+DI_t+(-DI_t)} $$
$$ ADX_t=RMA_n(DX)_t $$

- **RMA 以首 $n$ 筆有效資料 SMA 起始**，之後每次用 $[前值\times(n-1)+當期值]/n$ 遞迴；首個 ADX 是 $n$ 筆 DX 的算術平均。
- **訊號：** +DI 上穿 -DI 且 ADX ≥ 門檻 → 看漲；-DI 上穿 +DI 且 ADX ≥ 門檻 → 看跌。
- **重要：** ADX **本身只表示趨勢強度，不代表上漲或下跌**；方向由 DI 決定。初始結果需要較長暖機期。
""", "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx"),
    ]
    return data

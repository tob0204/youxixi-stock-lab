# 油洗洗 v1.0｜GitHub + Streamlit（無本機安裝）

## 功能

- 台股收盤後，只研究 **下一交易日**：假設隔天開盤可按建議限價進場、當日收盤離場（研究模型，非盤中真實成交驗證）。
- 十類技術指標：MA、EMA、KD、MACD、布林通道、RSI、斐波那契回調、一目均衡表、標準差、ADX。
- **20 種分開回測的訊號策略**：把「當天真正發生交叉」與「已持續一段時間的多空狀態」分開測；例如 KD 黃金交叉 ≠ K>D 的每一天。
- 同時比較 2 週、1 個月、3 個月、6 個月與 12 個月的歷史樣本，保留逐日 walk-forward 驗證。
- **點表格任意一列（尤其樣本數）即可在該表格下方展開全部樣本**，含訊號日、進場日、結果日、訊號收盤、隔日開盤、隔日收盤、扣成本後報酬及成功／失敗；可以下載完整 CSV，不用額外挑選指標。
- 單股名稱與代號、最新收盤價、下一日方向／參考限價；產業選單設在左側，海選結果位於右側「產業掃描結果」頁籤；每次最多分批 40 檔，以降低公開行情限流風險。
- 「技術指標公式與訊號規則」頁籤逐一公開全部公式、初值、參數、訊號定義及外部資料連結；「原始指標數值」可輸出所有歷史中間值。
- 修正先前產業海選因訊號名稱與舊式十指標清單不一致而被篩空的問題。
- 低於 160 根日 K 才視為未滿基本暖機資料；來源抓取失敗與資料太少分別顯示，**抓取失敗不等於股票不存在**。

## 直接更新現有 GitHub 與 Streamlit

1. 下載本 ZIP 並解壓縮，只需用網頁操作，不需安裝任何程式。
2. GitHub Repository 已建立好，不必再建一次。**先確認 Streamlit Settings → General → Main file path**：
   - 若指向 `youxixi_starter_v02/app.py`，請把全部 `.py`、`requirements.txt`、`README.md` 上傳到 GitHub 的 `youxixi_starter_v02/` 資料夾（與它的 `app.py` **同層**）。
   - 若指向 `app.py`，則把它們上傳到 GitHub **根目錄**。
3. GitHub 點 Commit changes。Streamlit Community Cloud 通常會隨 GitHub 的相同分支自動重新部署；回到原本網址確認版本是 `v1.0`。
4. 左側選股票，按「分析這檔股票」→ 首頁「看完整技術訊號比較、點選樣本數即可核對日期」→ **點其中一列** → 在表格下方直接看到逐筆結果。
5. 要海選，從左側「依產業海選」選官方產業並按掃描，切換右側「產業掃描結果」看結果。

**GitHub 網頁有時無法用 Upload files 覆蓋同名檔案**。可直接按鍵盤 `.` 打開 github.dev 網頁編輯器，依檔名覆蓋並 Commit；沒有必要裝 VS Code。

## 核心算法與參考來源

指標的數學式和策略的「訊號規則」不同；即使公式正確，策略規則也未必有預測能力。

| 類別 | 核對重點 | 參考 |
|---|---|---|
| MA | N 根收盤 SMA，短長交叉／排列分開 | [StockCharts MA](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential) |
| EMA | **v1.0 改採首 N 根收盤 SMA 作第一筆 EMA**，之後 α=2/(N+1) | [StockCharts MA](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential) |
| KD | 採臺灣市場慣用 RSV(9)，K、D 各以 50 起始的 1/3 遞迴；**不是** StockCharts SMA 平滑的 Fast/Slow KD。交叉、K>D 狀態、20/80 反轉各自獨立 | [臺灣 KD 算法](https://www.ezchart.com.tw/inds.php?IND=KD)、[StockCharts Stochastic](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stochastic-oscillator-fast-slow-and-full)、[Fidelity](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/fast-stochastic) |
| MACD | DIF=EMA12−EMA26，DEA=EMA9(DIF)，柱=DIF−DEA。交叉、零軸、柱正負分開 | [StockCharts MACD](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator) |
| 布林 | MA20 ± 2×母體標準差(ddof=0)，通道突破與回落為不同策略 | [StockCharts Bollinger](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/bollinger-bands) |
| RSI | 14 根 Wilder RMA；50 中線與 30/70 回復信號不同 | [StockCharts RSI](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi) |
| Fibonacci | 23.6/38.2/50/61.8/78.6；60 根可得高低點客製策略，不是唯一公定的買賣規則 | [StockCharts Fibonacci](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-annotation-tools/fibonacci-retracements) |
| 一目 | 9/26/52；雲層向未來畫 26 根，回測今天看得見的雲需 `shift(26)` | [StockCharts Ichimoku](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/ichimoku-cloud) |
| 標準差 | 母體標準差 ddof=0；只有 Z-score 及反轉條件才會產生方向 | [Pandas std](https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.std.html) |
| ADX | Wilder RMA；ADX 只顯示強弱，+DI/−DI 給多空方向 | [StockCharts ADX](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx) |

## 研究解讀與限制

- **狀態型訊號高度連續相關**：K>D 連續 20 天，不等於 20 個獨立金叉；此版仍把每天當成一個「隔日方向預測樣本」列出，但不能把筆數當成獨立統計檢定樣本數。
- 五個歷史訓練窗高度重疊、20 種規則又多重比較，最漂亮的樣本內勝率可能有選模偏誤；請觀察另一個「逐日前推驗證」結果。
- 買入／賣出為研究訊號，不是保證可賺的交易建議。賣出／看跌不表示可以放空。限價由歷史開盤跳空百分位數估算，並非預言明天開盤價。
- 交易成本目前示例採往返 **0.45%**，不是你的券商實際成本；流動性、價格限制、成交量及滑價尚未完整建模。
- **只有日 K，不是分鐘級當沖實際回測**；待後續加入盤中 OHLC 與交易所成交資格才適合做實單風控。選擇權未納入。
- 無法使用來源或最新日 K 未完成時，會顯示原因而非假造價格。Streamlit Cloud 的 Yahoo 行情與臺灣官方 API 可用性，需實際部署後驗證。

## 測試

已在隔離環境執行 `python -m unittest discover`：共 38 項離線測試通過，包含公式初值、20 種策略、樣本日期完整性、歷史未來資料防漏與原本錯誤的空值情境。**本次尚無 Streamlit Cloud 實際操作端到端測試**。

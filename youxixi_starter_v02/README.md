# 油洗洗｜台股量化訊號回測系統 v0.4（GitHub + Streamlit Cloud）

此版本供 **GitHub Repository → Streamlit Community Cloud → Chrome 瀏覽器** 使用。沒有 Windows 安裝檔，暫時不含選擇權試算。

## 直接部署到「你原本的 GitHub Repository」

1. 在電腦解壓本 ZIP；**不要上傳 ZIP 本身，也不要再包一層資料夾**。
2. 打開你已建立的 GitHub Repository，進入最上層（根目錄），點 **Add file → Upload files**，選擇以下檔案：
   - `app.py`
   - `stock_names.py`
   - `requirements.txt`
   - `README.md`
   - `test_stock_names.py`（可省略，建議保留）
3. GitHub 若提示檔名相同，以本版檔案**更新／覆蓋**原本的 `app.py`、`requirements.txt`；完成後按 **Commit changes**。不要留下另一個 `app_v04.py` 當主檔，或將檔案放在 ZIP 同名子資料夾內。
4. 打開原本 Streamlit Community Cloud 網站。若已部署同一個 Repository，GitHub commit 後 Streamlit 一般會自動更新，不需重建網站；如沒更新，可在 **Manage app** 檢查、重新啟動與錯誤日誌。
5. 若是第一次部署，在 https://share.streamlit.io/ → Create app → 填入 GitHub Repository 所有者／名稱；`Branch` 選 Repository 實際使用的分支（通常 `main`）；`Main file path` 填 **`app.py`** → Deploy。主檔不是 `streamlit_app.py`。

## 這個版本會做什麼？

- 用官方證交所、櫃買中心最新行情 API 取得「代號／當前中文名稱」，支援中文名稱搜尋與已知名單備援。
- 用 `yfinance` 取得選定股票的日線資料（歷史行情）。
- 示範 RSI(14) 50 穿越、KD(9,3,3) K/D 交叉、MACD(12,26,9) 交叉；統計 1/3/5/10/20 **交易日**後的方向命中率。
- 輸出一般樣本成功率、近期半衰期加權成功率、樣本數、Wilson 95% 區間和未扣交易成本的方向報酬。
- 回測完成後可下載 CSV，含中文名稱和代號；下載不會把畫面清空。
- **選擇權試算已移除**。沒有券商自動下單功能。

## 需要注意

- 日線收盤後才產生訊號，評估下一交易日開盤以後的方向，**不是當沖回測**。
- 勝率不能直接視為未來機率，也不是扣除交易成本後的正期望值。
- 尚缺 1 分鐘／5 分鐘成交資料、滑價、費稅、外樣本驗證；因此不應直接用於真實交易決策。
- 行情與中文名稱依賴第三方／官方網站，雲端若出現網路限制，可能會顯示資料取得失敗或有限離線備援清單。
- GitHub 與 Streamlit Cloud 若使用公開 Repo，所有人都能看程式。勿上傳券商帳密、API token、交易明細、身份資訊。

## 檔案架構（直接放 GitHub 最上層）

```text
你的 GitHub Repository/
├── app.py                # Streamlit 網頁與回測引擎
├── stock_names.py        # 中文名稱查詢
├── requirements.txt      # Streamlit 自動安裝的套件
├── test_stock_names.py   # 可選：名稱模組測試
└── README.md             # 本說明
```

### 錯誤排查

- `This file does not exist` → Main file path 請填 `app.py`，並確認 GitHub 根目錄的 `app.py` 存在。
- `This branch does not exist` → 到 GitHub Repository 頁面確認分支名稱，如 `main`，不要沿用預設的 `master`。
- `ModuleNotFoundError` → 確認 `requirements.txt` 與 `app.py` 在同一層，且 `stock_names.py` 有一起上傳。
- Yahoo Finance 回傳空資料 → 換個日期／代號或稍後再試；單一資料源可能有速率限制。

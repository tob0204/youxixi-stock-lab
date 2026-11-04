"""油洗洗股票中文名稱查詢（臺灣證交所與櫃買中心官方開放資料）。

名稱是方便使用者辨識的當前名稱，不是歷史基準日的 point-in-time 名稱。
API 可能暫停或異動，因此預留有限的離線備援，失敗時不捏造公司名稱。
"""
import re
import requests

TWSE_NAMES_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX_NAMES_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"

# 在官方 API 暫時不可用時，僅提供少數明確已知的代號／名稱。
FALLBACK_STOCKS = {
    "2330": ("台積電", "上市"),
    "2454": ("聯發科", "上市"),
    "3665": ("貿聯-KY", "上市"),
    "6139": ("亞翔", "上市"),
    "2303": ("聯電", "上市"),
    "3711": ("日月光投控", "上市"),
    "6239": ("力成", "上市"),
    "2449": ("京元電子", "上市"),
    "5347": ("世界", "上櫃"),
    "3443": ("創意", "上市"),
    "0056": ("元大高股息", "上市"),
    "00878": ("國泰永續高股息", "上市"),
    "00919": ("群益台灣精選高息", "上市"),
}


def parse_market_rows(rows, code_key, name_key, market):
    """將某市場官方 JSON 列表轉成代號到中文名稱的對照。"""
    found = {}
    if not isinstance(rows, list):
        raise ValueError("市場名稱資料格式並非列表")
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = str(row.get(code_key) or "").strip()
        name = str(row.get(name_key) or "").strip()
        if not re.fullmatch(r"\d{4,5}", code) or not name:
            continue
        found[code] = (name, market)
    return found


def fetch_stock_names(timeout_seconds=12):
    """返回 (code -> (name,market), errors)；部分市場失效亦不妨礙繼續使用。"""
    stocks = dict(FALLBACK_STOCKS)
    errors = []
    sources = [
        (TWSE_NAMES_URL, "Code", "Name", "上市"),
        (TPEX_NAMES_URL, "SecuritiesCompanyCode", "CompanyName", "上櫃"),
    ]
    for url, code_key, name_key, market in sources:
        try:
            response = requests.get(url, timeout=timeout_seconds, headers={"User-Agent": "YouxixiResearch/0.3"})
            response.raise_for_status()
            parsed = parse_market_rows(response.json(), code_key, name_key, market)
            if not parsed:
                raise ValueError("未解析到任何股票名稱")
            stocks.update(parsed)
        except (requests.RequestException, ValueError, TypeError) as exc:
            errors.append(f"{market}中文名稱服務暫不可用：{str(exc)[:90]}")
    return stocks, errors


def find_stock_name(code, names):
    """未知的代號不冒稱已查得名稱。"""
    clean = str(code).strip().upper().split(".")[0]
    return names.get(clean, ("中文名稱未取得", "市場未確認"))

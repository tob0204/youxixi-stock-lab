"""油洗洗：依 TWSE / TPEx 官方「產業別」查詢個股，不用猜測概念股。

如官方端點不可用，明確標記不完整；不把無法判定的個股隨意歸入產業。
"""
from __future__ import annotations
from collections import defaultdict
import re
import requests

TWSE_PROFILE = 'https://openapi.twse.com.tw/v1/opendata/t187ap03_L'
TPEX_PROFILE = 'https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O'

INDUSTRIES = {
    '01':'水泥工業', '02':'食品工業', '03':'塑膠工業', '04':'紡織纖維',
    '05':'電機機械', '06':'電器電纜', '08':'玻璃陶瓷', '09':'造紙工業',
    '10':'鋼鐵工業', '11':'橡膠工業', '12':'汽車工業', '14':'建材營造',
    '15':'航運業', '16':'觀光餐旅', '17':'金融保險', '18':'貿易百貨',
    '19':'綜合', '20':'其他', '21':'化學工業', '22':'生技醫療業',
    '23':'油電燃氣業', '24':'半導體業', '25':'電腦及週邊設備業',
    '26':'光電業', '27':'通信網路業', '28':'電子零組件業',
    '29':'電子通路業', '30':'資訊服務業', '31':'其他電子業',
    '32':'文化創意業', '33':'農業科技業', '35':'綠能環保',
    '36':'數位雲端', '37':'運動休閒', '38':'居家生活',
}

def _first(row, *keys):
    for k in keys:
        v=row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ''

def normalize_industry(raw):
    """收斂為兩位碼；未知/不屬股票的產業代碼不納入。"""
    val=str(raw or '').strip()
    if re.fullmatch(r'\d{1,2}',val):
        code=val.zfill(2)
        return code if code in INDUSTRIES else None
    return None

def parse_profiles(rows, market):
    if not isinstance(rows,list):
        raise ValueError('公司基本資料不是 JSON 列表')
    profiles={}
    for row in rows:
        if not isinstance(row,dict): continue
        code=_first(row,'公司代號','股票代號','CompanyCode','Code')
        if not re.fullmatch(r'[1-9]\d{3}',code): continue  # 普通股 4 碼；排除 ETF 和其他證券
        sector=normalize_industry(_first(row,'產業別','IndustryCode','產業類別'))
        name=_first(row,'公司簡稱','公司名稱','CompanyName','SecuritiesCompanyName')
        if not sector or not name: continue
        profiles[code]={'名稱':name,'市場':market,'產業代碼':sector,
                        '產業':INDUSTRIES[sector]}
    return profiles

def load_official_industries(timeout=12, getter=None):
    getter=getter or requests.get
    combined={}; errors=[]; sources=[]
    for market,url in [('上市',TWSE_PROFILE),('上櫃',TPEX_PROFILE)]:
        try:
            response=getter(url,timeout=timeout,headers={'User-Agent':'Mozilla/5.0 (Youxixi Research)'})
            response.raise_for_status()
            data=parse_profiles(response.json(),market)
            if not data: raise ValueError('無有效股票或產業欄位')
            combined.update(data); sources.append(market)
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            errors.append(f'{market}官方產業資料無法取得：{str(exc)[:95]}')
    return combined, sources, errors

def sectors_summary(profiles):
    grouped=defaultdict(list)
    for code,info in profiles.items():
        grouped[info['產業代碼']].append(code)
    return {sid:sorted(codes) for sid,codes in sorted(grouped.items()) if codes}

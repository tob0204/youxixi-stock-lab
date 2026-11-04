"""v0.9 核心回歸測試：沒有樣本不再崩潰，產業海選只掃指定公司。"""
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from sector_engine import parse_profiles, sectors_summary, load_official_industries
from forward_engine import predict_day
from audit_engine import build_ledger
from window_engine import WINDOW_LABELS

class TestV09(unittest.TestCase):
    def test_null_wilson_after_one_current_signal_does_not_crash(self):
        idx=pd.bdate_range('2024-01-01',periods=510)
        close=100 + np.arange(510)*.04
        prices=pd.DataFrame({'Open':close+.1,'Close':close,'High':close+1,
                            'Low':close-1,'Volume':50000},index=idx)
        signal=np.zeros(510,dtype=int); signal[-1]=1
        sig={'KD 隨機指標':pd.Series(signal,index=idx)}
        ledger=build_ledger(prices,sig,[1],.45)
        result=predict_day(prices,sig,ledger,509,WINDOW_LABELS,50,8,20,3,True)
        self.assertEqual(result['判斷'],'觀望')
        self.assertEqual(result['研究性候選']['符合限價樣本數'],0)

    def test_classification_uses_official_sector_code(self):
        profiles=parse_profiles([
            {'公司代號':'2330','公司簡稱':'台積電','產業別':'24'},
            {'公司代號':'2317','公司簡稱':'鴻海','產業別':'31'},
            {'公司代號':'0056','公司簡稱':'基金','產業別':'00'},
            {'公司代號':'9999','公司簡稱':'無資料','產業別':''},
        ],'上市')
        groups=sectors_summary(profiles)
        self.assertEqual(groups['24'],['2330'])
        self.assertEqual(groups['31'],['2317'])
        self.assertNotIn('0056',profiles)
        self.assertNotIn('9999',profiles)

    def test_fetch_partial_official_source_is_explicit(self):
        def get(url,**kw):
            if 'twse' in url:
                result=Mock();result.raise_for_status.return_value=None
                result.json.return_value=[{'公司代號':'2330','公司簡稱':'台積電','產業別':'24'}]
                return result
            raise ValueError('mock unavailable')
        profiles,sources,errors=load_official_industries(getter=get)
        self.assertEqual(list(profiles),['2330'])
        self.assertEqual(sources,['上市'])
        self.assertEqual(len(errors),1)

    def test_unknown_code_not_classified(self):
        self.assertEqual(parse_profiles([{'公司代號':'1234','公司簡稱':'測試','產業別':'99'}],'上市'),{})

if __name__=='__main__':unittest.main()

"""油洗洗 v0.7 多期間自動比較測試。"""
import unittest
import pandas as pd
import numpy as np
from window_engine import cutoff_for_window, compare_windows, recommend_across_windows, WINDOW_LABELS

class TestMultiWindow(unittest.TestCase):
    def test_periods(self):
        d=pd.Timestamp('2026-11-04')
        self.assertEqual(cutoff_for_window(d,'2週'),pd.Timestamp('2026-10-21'))
        self.assertEqual(cutoff_for_window(d,'3個月'),pd.Timestamp('2026-08-04'))
        self.assertEqual(len(WINDOW_LABELS),5)

    def test_2week_and_12months_count(self):
        days=pd.bdate_range('2025-01-01',periods=350)
        prices=pd.DataFrame({'Open':100.0,'Close':103.0},index=days)
        sig=pd.Series([1]*len(days),index=days)
        df=compare_windows(prices,{'MA':sig},[1],20,0.5)
        d2=df[(df['回測期間']=='2週')&(df['方向']=='看漲')].iloc[0]
        d12=df[(df['回測期間']=='12個月')&(df['方向']=='看漲')].iloc[0]
        self.assertLess(d2['訊號數'],d12['訊號數'])
        self.assertLessEqual(d2['訊號數'],11)
        self.assertEqual(d2['成本後成功次數'],d2['訊號數'])

    def mock_results(self,n2=2,rate2=50):
        rows=[]
        for label, n, rate in [('2週',n2,rate2),('1個月',4,75),('3個月',35,95),('6個月',65,92),('12個月',110,91)]:
            rows.append({'回測期間':label,'技術指標':'MA','方向':'看漲','預測交易日':1,
                         '訊號數':n,'成本後成功次數':round(n*rate/100),
                         '成本後命中率%（簡化）':rate,
                         '近期加權命中率%':96,'近期有效樣本數':12,
                         '平均方向報酬%（簡化扣成本）':1.2})
        return pd.DataFrame(rows)

    def test_3months_qualifies_2weeks_inconclusive(self):
        sig={'MA':pd.Series([0,1],index=pd.bdate_range('2026-11-03',periods=2))}
        status,tab,why,best=recommend_across_windows(sig,self.mock_results(),1,90,20,5,True)
        self.assertEqual(status,'買入')
        self.assertIn('兩週內僅 2 筆',why)
        self.assertIsNotNone(best)
        self.assertIn('3個月',tab['回測期間'].tolist())
        self.assertEqual(tab.loc[tab['回測期間']=='2週','評估'].iat[0],'樣本不足（2/20）')

    def test_2weeks_recent_deterioration_veto(self):
        sig={'MA':pd.Series([1])}
        status,_,reason,_=recommend_across_windows(sig,self.mock_results(n2=5,rate2=20),1,90,20,5,True)
        self.assertEqual(status,'觀望')

    def test_no_signal_no_recommendation(self):
        status,tab,_,_=recommend_across_windows({'MA':pd.Series([0])},self.mock_results(),1)
        self.assertEqual(status,'觀望')
        self.assertTrue((tab['評估']=='今天沒有新訊號').all())

    def test_threshold_changes_result(self):
        sig={'MA':pd.Series([1])}
        self.assertEqual(recommend_across_windows(sig,self.mock_results(),1,90)[0],'買入')
        self.assertEqual(recommend_across_windows(sig,self.mock_results(),1,99)[0],'觀望')

    def test_only_completed_signal_counted(self):
        days=pd.bdate_range('2025-01-01',periods=20)
        prices=pd.DataFrame({'Open':[100]*20,'Close':[105]*20},index=days)
        signals={'MA':pd.Series([0]*19+[1],index=days)}
        df=compare_windows(prices,signals,[1],5,0)
        self.assertTrue((df['訊號數']==0).all())

if __name__=='__main__':
    unittest.main()

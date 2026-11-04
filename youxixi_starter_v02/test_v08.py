"""油洗洗 v0.8：逐日前推、樣本可追溯、時間防洩漏等離線測試。"""
import unittest
import numpy as np
import pandas as pd
from audit_engine import build_ledger, select_ledger, dates_tooltip
from forward_engine import predict_day, walk_forward, tick_round
from window_engine import WINDOW_LABELS


class TestOneDayForecast(unittest.TestCase):
    def setUp(self):
        self.days=pd.bdate_range('2023-01-02',periods=760)
        x=np.arange(len(self.days))
        close=100+0.015*x+2*np.sin(x/6)
        self.prices=pd.DataFrame({'Open':close*(1+0.002*np.cos(x)),
            'High':close*1.02,'Low':close*.98,'Close':close,'Volume':10000},index=self.days)
        flags=np.zeros(len(self.days),dtype=int)
        flags[100::5]=1
        self.sig={'KD':pd.Series(flags,index=self.days)}
        self.ledger=build_ledger(self.prices,self.sig,[1],.45)

    def test_ledger_does_not_include_future_result(self):
        t=600
        chosen=select_ledger(self.ledger,'KD','看漲',1,self.days[t],'12個月')
        self.assertTrue((pd.to_datetime(chosen['結果日'])<=self.days[t]).all())
        self.assertTrue((pd.to_datetime(chosen['訊號日'])>=self.days[t]-pd.DateOffset(months=12)).all())

    def test_latest_cannot_see_tomorrow_actual_prices(self):
        T=600
        a=predict_day(self.prices,self.sig,self.ledger,T,WINDOW_LABELS,60,5,20,2)
        altered=self.prices.copy()
        altered.loc[altered.index[T+1:],'Open']*=11
        altered.loc[altered.index[T+1:],'Close']*=9
        ledger2=build_ledger(altered,self.sig,[1],.45)
        b=predict_day(altered,self.sig,ledger2,T,WINDOW_LABELS,60,5,20,2)
        self.assertEqual(a['判斷'],b['判斷'])
        ca=[(x['回測期間'],x['歷史估計限價'],x['成功次數']) for x in a['候選明細']]
        cb=[(x['回測期間'],x['歷史估計限價'],x['成功次數']) for x in b['候選明細']]
        self.assertEqual(ca,cb)

    def test_walk_forward_has_exact_previous_trading_day(self):
        f=walk_forward(self.prices,self.sig,self.ledger,self.days[510],WINDOW_LABELS,
                       60,5,20,2,True,.45)
        self.assertEqual(len(f),len(self.days)-510)
        self.assertEqual(f.iloc[0]['預測目標日'],str(self.days[510].date()))
        self.assertEqual(f.iloc[0]['訓練截止日'],str(self.days[509].date()))
        self.assertEqual(f.iloc[-1]['訓練截止日'],str(self.days[-2].date()))
        self.assertTrue((pd.to_datetime(f['訓練截止日'])<pd.to_datetime(f['預測目標日'])).all())

    def test_price_tick_round(self):
        self.assertEqual(tick_round(100.33,1),100.0)
        self.assertEqual(tick_round(100.33,-1),100.5)
        self.assertEqual(tick_round(49.99,1),49.95)

    def test_hover_lists_each_of_24_dates(self):
        days=pd.bdate_range('2026-09-01',periods=24)
        tab=pd.DataFrame([{'訊號日':str(d.date()),'結果日':str(d.date()),
                           '是否成功':i%2==0,'扣成本後報酬%':1 if i%2==0 else -2}
                          for i,d in enumerate(days)])
        report=dates_tooltip(tab,max_chars=12000)
        self.assertEqual(report.count('→'),24)
        self.assertEqual(report.count('成功'),12)
        self.assertEqual(report.count('失敗'),12)

    def test_no_event_means_wait(self):
        none={'KD':pd.Series(np.zeros(len(self.days),dtype=int),index=self.days)}
        p=predict_day(self.prices,none,build_ledger(self.prices,none,[1],.45),
                      len(self.days)-1,WINDOW_LABELS,90,20,20,5)
        self.assertEqual(p['判斷'],'觀望')
        self.assertIsNone(p['主要模型'])

if __name__=='__main__': unittest.main()

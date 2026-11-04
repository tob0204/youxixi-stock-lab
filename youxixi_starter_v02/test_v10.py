"""v1.0：公式、20 種訊號、記錄日期完整性及真正可點選的明細資料測試（離線）。"""
import unittest
import numpy as np
import pandas as pd
from indicators import _ema_sma_seed, DEFAULTS, NAMES, compute_indicators, make_signals
from formulas import formula_sections,signal_catalog
from audit_engine import build_ledger
from forward_engine import predict_day
from window_engine import WINDOW_LABELS


class TestV10(unittest.TestCase):
    def setUp(self):
        days=pd.bdate_range('2022-01-01',periods=600)
        rng=np.random.default_rng(123)
        close=100+np.cumsum(rng.normal(.1,1.5,len(days)))
        op=close+rng.normal(0,.5,len(days))
        high=np.maximum(op,close)+.5
        low=np.minimum(op,close)-.5
        self.p=pd.DataFrame({'Open':op,'High':high,'Low':low,'Close':close,'Volume':1000},index=days)

    def test_ema_initial_is_sma_and_next_step_is_recursive(self):
        e=_ema_sma_seed(pd.Series([1.,2,3,4,5]),3)
        self.assertTrue(np.isnan(e.iloc[1]))
        self.assertEqual(e.iloc[2],2)
        self.assertEqual(e.iloc[3],3)
        self.assertEqual(e.iloc[4],4)

    def test_signals_match_all_documented_rules(self):
        v=compute_indicators(self.p)
        signals=make_signals(v)
        self.assertEqual(len(NAMES),10)
        self.assertEqual(len(signals),20)
        self.assertEqual(list(signals),[x[0] for x in signal_catalog()])
        self.assertTrue(all(set(x.unique()).issubset({-1,0,1}) for x in signals.values()))
        self.assertTrue('KD｜K/D 黃金／死亡交叉' in signals)
        self.assertTrue('KD｜K＞D 多頭／K＜D 空頭（狀態）' in signals)

    def test_kd_cross_event_is_not_kd_state(self):
        v=compute_indicators(self.p)
        s=make_signals(v)
        event=s['KD｜K/D 黃金／死亡交叉']
        state=s['KD｜K＞D 多頭／K＜D 空頭（狀態）']
        self.assertGreater(state.ne(0).sum(),event.ne(0).sum())
        self.assertTrue(((event==1) & (state==-1)).sum()==0)
        self.assertTrue(((event==-1) & (state==1)).sum()==0)

    def test_history_ledger_contains_every_completed_entry(self):
        v=compute_indicators(self.p);s=make_signals(v)
        ledger=build_ledger(self.p,s,[1],.45)
        self.assertFalse(ledger.empty)
        p=predict_day(self.p,s,ledger,len(self.p)-1,WINDOW_LABELS,50,8,20,3,True)
        self.assertTrue(p['候選明細'])
        for candidate in p['候選明細']:
            selected=candidate['成交樣本明細']
            self.assertEqual(len(selected),candidate['符合限價樣本數'])
            if selected.empty: continue
            self.assertEqual(int(selected['是否成功'].sum()),candidate['成功次數'])
            self.assertTrue((pd.to_datetime(selected['結果日'])<=self.p.index[-1]).all())

    def test_history_calcs_do_not_see_future(self):
        current=compute_indicators(self.p.iloc[:360])
        future=self.p.copy()
        future.loc[future.index[360:],['High','Low','Close']]*=2
        whole=compute_indicators(future)
        for col in current.columns:
            pd.testing.assert_series_equal(current[col],whole.loc[current.index,col],check_names=False)

if __name__=='__main__':unittest.main()

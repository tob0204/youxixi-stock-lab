"""油洗洗 v0.6 自動化測試。GitHub 上可檢視，不需要使用者本機安裝。"""
import math
import unittest

import numpy as np
import pandas as pd

from indicators import DEFAULTS, NAMES, compute_indicators, make_signals
from formulas import formula_sections
from signal_engine import decide_latest, evaluate, wilson_95


class TestIndicators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        days = pd.bdate_range('2024-01-01', periods=340)
        x = np.arange(len(days))
        close = 100 + 0.025 * x + 8 * np.sin(x / 10) + 2 * np.sin(x / 2)
        cls.prices = pd.DataFrame({
            'Open': close + 0.15 * np.sin(x),
            'High': close + 1.6, 'Low': close - 1.8,
            'Close': close, 'Volume': 10000 + x,
        }, index=days)

    def test_ten_indicators_signals_and_formulas(self):
        values = compute_indicators(self.prices)
        signals = make_signals(values)
        self.assertEqual(len(signals), 10)
        self.assertEqual(len(formula_sections(DEFAULTS)), 10)
        self.assertEqual(list(signals), NAMES)
        for signal in signals.values():
            self.assertTrue(signal.isin([-1, 0, 1]).all())
            self.assertEqual(len(signal), len(self.prices))
        self.assertTrue(values['MA短'].notna().sum() > 100)
        self.assertTrue(values['雲A(當日可見)'].notna().sum() > 100)
        self.assertTrue(values['ADX'].notna().sum() > 100)

    def test_no_future_price_leakage(self):
        before = compute_indicators(self.prices.iloc[:250])
        after = self.prices.copy()
        after.loc[after.index[250:], ['Open', 'High', 'Low', 'Close']] *= 100
        full = compute_indicators(after)
        for field in before.columns:
            pd.testing.assert_series_equal(before[field], full[field].iloc[:250], check_names=False)

    def test_rsi_constant_prices_neutral(self):
        p = self.prices.copy()
        p[['Open', 'High', 'Low', 'Close']] = 100.0
        vals = compute_indicators(p)
        self.assertAlmostEqual(vals['RSI'].dropna().iloc[-1], 50.0)
        self.assertAlmostEqual(vals['布林σ'].dropna().iloc[-1], 0.0)


class TestBacktest(unittest.TestCase):
    def test_entry_next_open_exit_horizon_close(self):
        days = pd.bdate_range('2025-01-01', periods=6)
        prices = pd.DataFrame({
            'Open': [100, 101, 101, 100, 104, 100],
            'Close': [100, 110, 100, 108, 100, 100],
        }, index=days)
        signals = {'demo': pd.Series([1, 0, -1, 0, 0, 1], index=days)}
        results = evaluate(prices, signals, days[0], [1], 10, 0.0)
        buy = results[(results['方向']=='看漲')].iloc[0]
        sell = results[(results['方向']=='看跌')].iloc[0]
        self.assertEqual(buy['訊號數'], 1)  # 最後一根訊號未完成，不能偷看未來
        self.assertEqual(buy['成本後成功次數'], 1)  # t=0進場 open[1]=101，出場 close[1]=110
        self.assertEqual(sell['訊號數'], 1)  # t=2，下日開 100，收 108，做空代理失敗
        self.assertEqual(sell['成本後成功次數'], 0)

    def test_cost_reduces_win_rate(self):
        days = pd.bdate_range('2025-01-01', periods=4)
        prices = pd.DataFrame({'Open':[100,100,100,100], 'Close':[100,100.2,100,100]}, index=days)
        s = {'demo':pd.Series([1,0,0,0], index=days)}
        gross = evaluate(prices,s,days[0],[1],20,0)
        net = evaluate(prices,s,days[0],[1],20,0.5)
        self.assertEqual(gross.loc[0,'成本後成功次數'],1)
        self.assertEqual(net.loc[0,'成本後成功次數'],0)

    def test_wilson_bounds(self):
        low, high = wilson_95(20,20)
        self.assertLess(low, 1.0)
        self.assertEqual(high, 1.0)
        self.assertTrue(math.isnan(wilson_95(0,0)[0]))


class TestLatestDecision(unittest.TestCase):
    def metrics(self, name, direction, n, rate, recent=95, effective=12):
        return {
            '技術指標':name, '方向':direction, '預測交易日':1,
            '訊號數':n, '成本後命中率%（簡化）':rate,
            '近期加權命中率%':recent, '近期有效樣本數':effective,
        }

    def dec(self, latest, rows, **kwargs):
        signals={k:pd.Series([v],index=[pd.Timestamp('2026-11-04')]) for k,v in latest.items()}
        return decide_latest(signals,pd.DataFrame(rows),1,**kwargs)

    def test_buy_gt_90(self):
        status,tbl,_=self.dec({'MA':1},[self.metrics('MA','看漲',30,93)])
        self.assertEqual(status,'買入')
        self.assertEqual(tbl.iloc[0]['符合買賣門檻'],'是')

    def test_sell_gt_90(self):
        status,_,_=self.dec({'MACD':-1},[self.metrics('MACD','看跌',40,92)])
        self.assertEqual(status,'賣出')

    def test_exact_90_does_not_trigger(self):
        status,_,_=self.dec({'MA':1},[self.metrics('MA','看漲',40,90)])
        self.assertEqual(status,'觀望')

    def test_threshold_can_be_changed(self):
        rows=[self.metrics('MA','看漲',40,81,85)]
        self.assertEqual(self.dec({'MA':1},rows,threshold_pct=90)[0],'觀望')
        self.assertEqual(self.dec({'MA':1},rows,threshold_pct=80)[0],'買入')

    def test_no_event_wait(self):
        self.assertEqual(self.dec({'MA':0},[self.metrics('MA','看漲',70,99)])[0],'觀望')

    def test_insufficient_sample_wait(self):
        self.assertEqual(self.dec({'MA':1},[self.metrics('MA','看漲',4,100)])[0],'觀望')

    def test_recent_failure_wait(self):
        rows=[self.metrics('MA','看漲',40,95,50,12)]
        self.assertEqual(self.dec({'MA':1},rows)[0],'觀望')
        self.assertEqual(self.dec({'MA':1},rows,require_recent=False)[0],'買入')

    def test_effective_sample_wait(self):
        rows=[self.metrics('MA','看漲',40,99,99,3)]
        self.assertEqual(self.dec({'MA':1},rows)[0],'觀望')

    def test_conflict_wait(self):
        rows=[self.metrics('MA','看漲',40,99),self.metrics('KD','看跌',40,99)]
        status, _, reason = self.dec({'MA':1,'KD':-1},rows)
        self.assertEqual(status,'觀望')
        self.assertIn('多空',reason)

    def test_stale_data_wait(self):
        rows=[self.metrics('MA','看漲',40,99)]
        self.assertEqual(self.dec({'MA':1},rows,allow_signal=False)[0],'觀望')


if __name__ == '__main__':
    unittest.main()

import unittest
from risk import check_long_txo, long_option_pnl, monthly_account_action


class RiskTests(unittest.TestCase):
    def test_1_contract_budget(self):
        r = check_long_txo(option_premium_points=20, contracts=1,
                           cash_available=10000, single_trade_limit=1500,
                           fee_per_side_per_contract=30)
        self.assertEqual(r.premium_paid, 1000)
        self.assertEqual(r.reserved_cash, 1060)
        self.assertTrue(r.affordable)
        self.assertTrue(r.within_trade_limit)

    def test_reject_over_trading_limit(self):
        r = check_long_txo(option_premium_points=100, contracts=1,
                           cash_available=10000, single_trade_limit=1500)
        self.assertTrue(r.affordable)
        self.assertFalse(r.within_trade_limit)

    def test_reject_over_monthly_cash(self):
        r = check_long_txo(option_premium_points=300, contracts=1,
                           cash_available=10000, single_trade_limit=20000)
        self.assertFalse(r.affordable)

    def test_profit_and_full_loss(self):
        self.assertEqual(long_option_pnl(entry_points=20, exit_points=40, contracts=1, fee_per_side_per_contract=30), 940)
        self.assertEqual(long_option_pnl(entry_points=20, exit_points=0, contracts=1, fee_per_side_per_contract=0), -1000)

    def test_sweep_only_above_trading_cap(self):
        self.assertEqual(monthly_account_action(13400), (3400, 10000))
        self.assertEqual(monthly_account_action(8200), (0, 8200))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            check_long_txo(option_premium_points=-2, contracts=1,
                           cash_available=10000, single_trade_limit=1500)


if __name__ == '__main__':
    unittest.main()

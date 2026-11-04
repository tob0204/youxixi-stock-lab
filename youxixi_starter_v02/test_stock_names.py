import unittest
from unittest.mock import patch
from stock_names import parse_market_rows, fetch_stock_names, find_stock_name


class StockNamesTest(unittest.TestCase):
    def test_twse_name(self):
        self.assertEqual(parse_market_rows([{"Code": "2454", "Name": "聯發科"}], "Code", "Name", "上市")["2454"], ("聯發科", "上市"))

    def test_tpex_name(self):
        self.assertEqual(parse_market_rows([{"SecuritiesCompanyCode": "5347", "CompanyName": "世界"}], "SecuritiesCompanyCode", "CompanyName", "上櫃")["5347"], ("世界", "上櫃"))

    def test_filter_invalid_rows(self):
        self.assertEqual(parse_market_rows([{"Code": "abc", "Name": "x"}, {"Code": "9999", "Name": ""}], "Code", "Name", "上市"), {})

    def test_both_market_services_unavailable(self):
        from requests import Timeout
        with patch('stock_names.requests.get', side_effect=Timeout('simulated timeout')):
            stocks, errors = fetch_stock_names()
        self.assertEqual(stocks['2330'], ('台積電', '上市'))
        self.assertEqual(len(errors), 2)

    def test_known_and_unknown_names(self):
        self.assertEqual(find_stock_name('2454.TW', {"2454": ("聯發科", "上市")}), ("聯發科", "上市"))
        self.assertEqual(find_stock_name('9999', {}), ("中文名稱未取得", "市場未確認"))

    def test_one_market_is_down(self):
        from requests import Timeout
        from unittest.mock import Mock
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = [{"SecuritiesCompanyCode":"5347", "CompanyName":"世界"}]
        with patch('stock_names.requests.get', side_effect=[Timeout('no response'), response]):
            stocks, errors = fetch_stock_names()
        self.assertEqual(stocks['5347'], ('世界','上櫃'))
        self.assertEqual(len(errors),1)
        self.assertEqual(stocks['2454'], ('聯發科','上市'))


if __name__ == '__main__':
    unittest.main()

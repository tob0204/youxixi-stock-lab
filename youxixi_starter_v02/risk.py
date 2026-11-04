"""油洗洗：僅買進臺指選擇權（TXO）時的資金預算檢查。

這不是選擇權定價或即時行情引擎；不支援賣方、組合保證金、期貨。
"""
from dataclasses import dataclass

TXO_NTD_PER_POINT = 50


@dataclass(frozen=True)
class BudgetCheck:
    premium_paid: float
    estimated_round_trip_fees: float
    reserved_cash: float
    budget_remaining: float
    single_trade_limit: float
    affordable: bool
    within_trade_limit: bool


def check_long_txo(
    *,
    option_premium_points: float,
    contracts: int,
    cash_available: float,
    single_trade_limit: float,
    fee_per_side_per_contract: float = 0,
) -> BudgetCheck:
    """按 1 點=NT$50 計算 TXO 買方現金投入；雙邊費用採保守預留。

    僅檢查開倉的現金風險，不代表選擇權有獲利期望或能順利平倉。
    """
    if option_premium_points <= 0:
        raise ValueError("權利金點數必須大於 0")
    if type(contracts) is not int or contracts <= 0:
        raise ValueError("口數必須是正整數")
    if cash_available < 0 or single_trade_limit <= 0 or fee_per_side_per_contract < 0:
        raise ValueError("可用現金不可為負數；限額須為正；費用不可為負數")
    premium = option_premium_points * TXO_NTD_PER_POINT * contracts
    fees = 2 * fee_per_side_per_contract * contracts
    required = premium + fees
    return BudgetCheck(
        premium_paid=premium,
        estimated_round_trip_fees=fees,
        reserved_cash=required,
        budget_remaining=cash_available - required,
        single_trade_limit=single_trade_limit,
        affordable=required <= cash_available,
        within_trade_limit=required <= single_trade_limit,
    )


def long_option_pnl(
    *, entry_points: float, exit_points: float, contracts: int,
    fee_per_side_per_contract: float = 0,
) -> float:
    """TXO 買進→賣出平倉的示例現金損益；只採輸入的買賣成交權利金。"""
    if entry_points < 0 or exit_points < 0 or type(contracts) is not int or contracts <= 0:
        raise ValueError("權利金不可小於 0；口數需為正整數")
    if fee_per_side_per_contract < 0:
        raise ValueError("費用不可小於 0")
    return (exit_points - entry_points) * TXO_NTD_PER_POINT * contracts - (2 * fee_per_side_per_contract * contracts)


def monthly_account_action(account_cash: float, trading_cap: float = 10000) -> tuple[float, float]:
    """將超過上限的現金全數轉出，不會從外部資金補洞。回傳(轉出, 保留)。"""
    if account_cash < 0 or trading_cap <= 0:
        raise ValueError("餘額不可小於 0；上限須為正")
    withdrawal = max(0.0, account_cash - trading_cap)
    return withdrawal, account_cash - withdrawal

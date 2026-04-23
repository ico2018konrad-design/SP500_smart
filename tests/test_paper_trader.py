"""Tests for PaperTrader capital accounting.

BUG #1 regression tests: verify that:
- Opening a position debits the full cost (shares * price + commission)
- Closing at the same price credits proceeds and leaves capital = initial - 2*commission - slippage
- Phantom profits are NOT created
"""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.execution.paper_trader import PaperTrader, COMMISSION_PCT, SLIPPAGE_PCT
from src.signals.signal_types import Signal, SignalDirection


def _make_signal(price: float = 100.0, symbol: str = "SPY") -> Signal:
    """Helper: create a LONG signal at given price."""
    return Signal(
        direction=SignalDirection.LONG,
        symbol=symbol,
        entry_price=price,
        stop_price=price * 0.985,
        target1=price * 1.02,
        target2=price * 1.045,
        target3=price * 1.08,
        timestamp=datetime.now(),
        regime_score=9,
        regime="BULL",
    )


def _patch_price(pt: PaperTrader, price: float):
    """Patch get_current_price to return a fixed price."""
    pt.get_current_price = lambda symbol: price


class TestPaperTraderCapitalAccounting:
    """Verify no phantom profit is created on open/close at same price."""

    def test_open_debits_full_cost(self):
        """Opening a position must reduce capital by shares*price + commission."""
        pt = PaperTrader(starting_capital=10_000.0)
        initial = pt.capital

        _patch_price(pt, 100.0)
        signal = _make_signal(price=100.0)

        # Set allocated capital so position sizing gives predictable result
        pos = pt.execute_signal(signal, allocated_capital=10_000.0)

        assert pos is not None, "Position should be opened"
        # Capital must be LESS than initial
        assert pt.capital < initial, "Capital must decrease after opening position"

        shares = pos.shares
        fill_price = pos.entry_price  # entry_price includes slippage
        commission = shares * fill_price * COMMISSION_PCT
        expected_capital = initial - shares * fill_price - commission
        assert abs(pt.capital - expected_capital) < 1.0, (
            f"Capital after open: {pt.capital:.2f}, expected: {expected_capital:.2f}"
        )

    def test_close_at_same_price_no_phantom_profit(self):
        """Close at same price → capital = initial - 2*commission - slippage_round_trip."""
        pt = PaperTrader(starting_capital=10_000.0)
        initial = pt.capital

        entry_price = 100.0
        _patch_price(pt, entry_price)

        signal = _make_signal(price=entry_price)
        pos = pt.execute_signal(signal, allocated_capital=10_000.0)
        assert pos is not None

        # Close at same underlying price
        _patch_price(pt, entry_price)
        pnl = pt.close_position(pos.position_id, reason="test")

        assert pnl is not None

        # Capital should be close to initial minus round-trip transaction costs
        # (2 commissions + 2 slippage costs absorbed in fill prices)
        # It should definitely be LESS than initial (no phantom profit)
        assert pt.capital <= initial, (
            f"Capital after flat round-trip should not exceed initial. "
            f"Got {pt.capital:.2f}, initial was {initial:.2f}."
        )

        # And it shouldn't be wildly below (e.g. not less than 99.5% of initial for small positions)
        assert pt.capital > initial * 0.98, (
            f"Capital loss on flat round-trip is unreasonably large: {pt.capital:.2f}"
        )

    def test_close_at_profit_increases_capital(self):
        """Close at profit → capital > initial - costs."""
        pt = PaperTrader(starting_capital=10_000.0)
        initial = pt.capital

        _patch_price(pt, 100.0)
        signal = _make_signal(price=100.0)
        pos = pt.execute_signal(signal, allocated_capital=10_000.0)
        assert pos is not None

        # Close at 5% gain
        _patch_price(pt, 105.0)
        pnl = pt.close_position(pos.position_id, reason="profit", current_prices={"SPY": 105.0})
        assert pnl is not None
        assert pnl > 0, "PnL should be positive"
        # Final capital should reflect the gain
        assert pt.capital > initial * 0.99, "Capital should not have dropped on profitable trade"

    def test_close_at_loss_decreases_capital(self):
        """Close at loss → capital < initial."""
        pt = PaperTrader(starting_capital=10_000.0)
        initial = pt.capital

        _patch_price(pt, 100.0)
        signal = _make_signal(price=100.0)
        pos = pt.execute_signal(signal, allocated_capital=10_000.0)
        assert pos is not None

        # Close at 2% loss
        _patch_price(pt, 98.0)
        pt.close_position(pos.position_id, reason="stop", current_prices={"SPY": 98.0})
        assert pt.capital < initial, "Capital should decrease on losing trade"

    def test_get_equity_includes_open_position_value(self):
        """get_equity() should return cash + mark-to-market of open positions."""
        pt = PaperTrader(starting_capital=10_000.0)

        _patch_price(pt, 100.0)
        signal = _make_signal(price=100.0)
        pos = pt.execute_signal(signal, allocated_capital=10_000.0)
        assert pos is not None

        # With open position, equity should approximately equal initial capital
        # (minus transaction costs, approximately)
        equity = pt.get_equity()
        assert equity > pt.capital, "Equity should be > cash when position is open"
        assert abs(equity - 10_000.0) < 10_000.0 * 0.05, (
            f"Equity {equity:.2f} too far from initial {10_000.0:.2f}"
        )

    def test_multiple_open_close_cycles_no_phantom_growth(self):
        """After N round-trips at flat prices, capital should not grow."""
        pt = PaperTrader(starting_capital=10_000.0)
        initial = pt.capital

        for _ in range(3):
            _patch_price(pt, 100.0)
            signal = _make_signal(price=100.0)
            pos = pt.execute_signal(signal, allocated_capital=pt.capital)
            if pos is None:
                break
            _patch_price(pt, 100.0)
            pt.close_position(pos.position_id, reason="flat", current_prices={"SPY": 100.0})

        assert pt.capital <= initial, (
            f"Capital grew after flat round-trips: {pt.capital:.2f} > {initial:.2f}"
        )

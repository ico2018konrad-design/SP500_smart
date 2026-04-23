"""Tests for anti-martingale scaler."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from src.positions.position_manager import PositionManager
from src.positions.anti_martingale_scaler import AntiMartingaleScaler
from src.signals.signal_types import Signal, SignalDirection


def make_signal(symbol="SPY", direction=SignalDirection.LONG) -> Signal:
    return Signal(
        direction=direction,
        symbol=symbol,
        entry_price=400.0,
        stop_price=394.0,
        target1=408.0,
        target2=418.0,
        target3=432.0,
        timestamp=datetime.now(),
        regime_score=9,
        regime="BULL",
    )


class TestAntiMartingaleScaler:
    def test_first_position_allowed(self):
        pm = PositionManager()
        scaler = AntiMartingaleScaler()
        can_scale, reason = scaler.can_scale_in(pm, "SPY", "LONG")
        assert can_scale is True
        assert "First position" in reason

    def test_no_scale_when_previous_not_profitable(self):
        pm = PositionManager()
        scaler = AntiMartingaleScaler(profit_threshold=0.01)

        # Add first position (at current price = no profit)
        pos = pm.add_position("SPY", "LONG", 400.0, 10, 394.0, 408.0, 418.0, 432.0)
        pos.current_price = 400.0  # no gain

        can_scale, reason = scaler.can_scale_in(pm, "SPY", "LONG")
        assert can_scale is False
        assert "not at" in reason or "anti-martingale" in reason.lower() or "profit" in reason.lower()

    def test_scale_in_when_profitable(self):
        pm = PositionManager()
        scaler = AntiMartingaleScaler(profit_threshold=0.01, min_hours_between=0)

        # Add first position with +2% profit
        pos = pm.add_position("SPY", "LONG", 400.0, 10, 394.0, 408.0, 418.0, 432.0)
        pos.current_price = 408.0  # +2% profit

        can_scale, reason = scaler.can_scale_in(pm, "SPY", "LONG")
        assert can_scale is True

    def test_never_scale_into_losing_position(self):
        pm = PositionManager()
        scaler = AntiMartingaleScaler(profit_threshold=0.01)

        pos = pm.add_position("SPY", "LONG", 400.0, 10, 394.0, 408.0, 418.0, 432.0)
        pos.current_price = 390.0  # losing -2.5%

        can_scale, reason = scaler.can_scale_in(pm, "SPY", "LONG")
        assert can_scale is False

    def test_max_positions_not_exceeded(self):
        pm = PositionManager(max_positions=2)
        scaler = AntiMartingaleScaler()

        pm.add_position("SPY", "LONG", 400.0, 5, 394.0, 408.0, 418.0, 432.0)
        pm.add_position("SPY", "LONG", 408.0, 5, 404.0, 416.0, 426.0, 440.0)

        can_scale, reason = scaler.can_scale_in(pm, "SPY", "LONG")
        assert can_scale is False
        assert "max" in reason.lower() or "capacity" in reason.lower()

    def test_stop_moves_to_be_on_scale_in(self):
        pm = PositionManager()
        scaler = AntiMartingaleScaler(profit_threshold=0.01, min_hours_between=0)

        pos1 = pm.add_position("SPY", "LONG", 400.0, 10, 394.0, 408.0, 418.0, 432.0)
        pos1.current_price = 408.0  # +2% profitable

        signal = make_signal()
        new_pos = scaler.execute_scale_in(
            pm, signal, capital=5000.0, current_prices={"SPY": 408.0}
        )

        assert new_pos is not None
        # Previous position stop should move to break-even
        assert pos1.breakeven_stop is True
        assert pos1.stop_price == pos1.entry_price

"""Tests for position manager."""
import pytest
from datetime import datetime

from src.positions.position_manager import Position, PositionManager, PositionStatus


class TestPositionManager:
    def test_can_add_position(self):
        pm = PositionManager(max_positions=6)
        assert pm.can_add_position() is True

    def test_add_position(self):
        pm = PositionManager()
        pos = pm.add_position(
            symbol="SPY",
            direction="LONG",
            entry_price=400.0,
            shares=10,
            stop_price=394.0,
            target1=408.0,
            target2=418.0,
            target3=432.0,
        )
        assert pos is not None
        assert pos.symbol == "SPY"
        assert pos.status == PositionStatus.OPEN
        assert pm.get_position_count() == 1

    def test_max_positions_enforced(self):
        pm = PositionManager(max_positions=2)
        for _ in range(2):
            pm.add_position("SPY", "LONG", 400.0, 5, 394.0, 408.0, 418.0, 432.0)

        assert pm.can_add_position() is False
        extra = pm.add_position("SPY", "LONG", 400.0, 5, 394.0, 408.0, 418.0, 432.0)
        assert extra is None

    def test_close_position(self):
        pm = PositionManager()
        pos = pm.add_position("SPY", "LONG", 400.0, 10, 394.0, 408.0, 418.0, 432.0)
        pnl = pm.close_position(pos.position_id, 420.0)
        assert pnl is not None
        assert pnl > 0  # should be profit
        assert pos.status == PositionStatus.CLOSED

    def test_flat_all(self):
        pm = PositionManager()
        for _ in range(3):
            pm.add_position("SPY", "LONG", 400.0, 5, 394.0, 408.0, 418.0, 432.0)
        pm.flat_all({"SPY": 410.0})
        assert pm.get_position_count() == 0

    def test_unrealized_pnl(self):
        pm = PositionManager()
        pos = pm.add_position("SPY", "LONG", 400.0, 10, 394.0, 408.0, 418.0, 432.0)
        pos.update_price(420.0)
        assert abs(pos.unrealized_pnl_pct - 0.05) < 0.001  # 5% gain

    def test_move_stop_to_breakeven(self):
        pm = PositionManager()
        pos = pm.add_position("SPY", "LONG", 400.0, 10, 394.0, 408.0, 418.0, 432.0)
        pm.move_stop_to_breakeven(pos.position_id)
        assert pos.stop_price == pos.entry_price
        assert pos.breakeven_stop is True

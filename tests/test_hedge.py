"""Tests for hedge modules."""
import pytest
from datetime import datetime

from src.hedge.baseline_hedge import BaselineHedge, HedgeStatus
from src.hedge.reactive_hedge import ReactiveHedge
from src.hedge.panic_hedge import PanicHedge, PanicTrigger


class TestBaselineHedge:
    def test_mini_mode_uses_sh(self):
        hedge = BaselineHedge(mode="mini")
        status = hedge.get_hedge_allocation(5000.0)
        assert status.hedge_instrument == "SH"
        assert abs(status.hedge_pct - 0.10) < 0.001

    def test_mini_mode_10pct(self):
        hedge = BaselineHedge(mode="mini")
        status = hedge.get_hedge_allocation(10000.0)
        assert abs(status.hedge_value - 1000.0) < 1.0

    def test_full_mode_below_threshold_uses_sh(self):
        hedge = BaselineHedge(mode="full")
        status = hedge.get_hedge_allocation(10000.0)  # below 15k threshold
        assert status.hedge_instrument == "SH"

    def test_sh_shares_calculation(self):
        hedge = BaselineHedge()
        shares = hedge.get_sh_shares(capital=5000.0, sh_price=50.0)
        # 10% of 5000 = 500 / 50 = 10 shares
        assert shares == 10

    def test_rebalancing_needed_outside_tolerance(self):
        hedge = BaselineHedge()
        assert hedge.needs_rebalancing(0.05, 0.10) is True  # 5% off target

    def test_no_rebalancing_within_tolerance(self):
        hedge = BaselineHedge()
        assert hedge.needs_rebalancing(0.095, 0.10) is False  # within 2%


class TestReactiveHedge:
    def test_triggers_on_spy_drop_and_vix_spike(self):
        rh = ReactiveHedge()
        action = rh.check_triggers(
            spy_intraday_return=-0.015,  # -1.5%
            vix_intraday_change=0.12,    # +12%
            current_regime="BULL",
            previous_regime="BULL",
        )
        assert action is not None
        assert action.instrument == "SPXS"

    def test_no_trigger_on_small_move(self):
        rh = ReactiveHedge()
        action = rh.check_triggers(
            spy_intraday_return=-0.005,  # only -0.5%
            vix_intraday_change=0.05,    # only +5%
            current_regime="BULL",
            previous_regime="BULL",
        )
        assert action is None

    def test_triggers_on_regime_downgrade(self):
        rh = ReactiveHedge()
        action = rh.check_triggers(
            spy_intraday_return=-0.005,
            vix_intraday_change=0.05,
            current_regime="CHOP",
            previous_regime="BULL",  # downgrade
        )
        assert action is not None
        assert action.instrument == "SH"


class TestPanicHedge:
    def test_vix_trigger(self):
        ph = PanicHedge(vix_threshold=35.0)
        trigger = ph.check_triggers(vix=38.0)
        assert trigger is not None
        assert trigger.trigger_type == "vix_spike"

    def test_no_trigger_normal_vix(self):
        ph = PanicHedge(vix_threshold=35.0)
        trigger = ph.check_triggers(vix=25.0)
        assert trigger is None

    def test_spy_crash_trigger(self):
        ph = PanicHedge(spy_2session_drop=-0.05)
        trigger = ph.check_triggers(vix=25.0, spy_2session_return=-0.06)
        assert trigger is not None
        assert trigger.trigger_type == "spy_crash"

    def test_panic_mode_requires_manual_unlock(self):
        ph = PanicHedge()
        trigger = PanicTrigger(
            trigger_type="vix_spike",
            trigger_value=40.0,
            threshold=35.0,
            timestamp=datetime.now(),
            description="Test",
        )
        status = ph.activate(trigger)
        assert status.is_active is True
        assert status.manually_locked is True
        assert not status.can_deactivate()

    def test_panic_allocations(self):
        ph = PanicHedge(spxs_pct=0.25, vxx_pct=0.03)
        trigger = PanicTrigger("vix_spike", 40.0, 35.0, datetime.now(), "Test")
        ph.activate(trigger)
        alloc = ph.get_allocations(5000.0)
        assert abs(alloc["SPXS"] - 1250.0) < 1.0  # 25% of 5000
        assert abs(alloc["VXX"] - 150.0) < 1.0    # 3% of 5000

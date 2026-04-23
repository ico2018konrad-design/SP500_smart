"""Tests for risk management."""
import pytest
from src.risk.circuit_breakers import CircuitBreakers, CircuitBreakerStatus
from src.risk.dynamic_risk_per_trade import DynamicRiskCalculator
from src.risk.atr_position_sizer import ATRPositionSizer
from src.risk.kill_switch import KillSwitch
from src.regime.regime_types import Regime
import tempfile
import os


class TestCircuitBreakers:
    def test_no_halt_normal_conditions(self):
        cb = CircuitBreakers()
        status = cb.update(current_equity=5000.0)
        assert not status.is_halted()

    def test_daily_halt_triggered(self):
        cb = CircuitBreakers(daily_limit=-0.03)
        cb._daily_start_equity = 5000.0
        # Simulate -4% loss
        status = cb.update(current_equity=4800.0)
        assert status.daily_halt is True
        assert not cb.can_trade()

    def test_consecutive_losses_trigger_cooldown(self):
        cb = CircuitBreakers(max_consecutive_losses=3)
        cb._daily_start_equity = 5000.0
        # 3 consecutive losses
        for _ in range(3):
            cb.update(current_equity=4990.0, closed_trade_win=False)
        assert not cb.can_trade()

    def test_consecutive_wins_reset_count(self):
        cb = CircuitBreakers()
        cb._daily_start_equity = 5000.0
        cb.status.consecutive_losses = 2
        cb.update(current_equity=5010.0, closed_trade_win=True)
        assert cb.status.consecutive_losses == 0

    def test_vix_flatten_threshold(self):
        cb = CircuitBreakers(vix_flatten_threshold=40.0)
        assert cb.vix_flatten_required(41.0) is True
        assert cb.vix_flatten_required(39.0) is False

    def test_flash_crash_detection(self):
        cb = CircuitBreakers()
        assert cb.flash_crash_detected(-0.025) is True  # -2.5% in 1 hour
        assert cb.flash_crash_detected(-0.015) is False  # only -1.5%


class TestDynamicRisk:
    def test_regime_determines_base_risk(self):
        calc = DynamicRiskCalculator()
        strong_bull_risk = calc.calculate(Regime.STRONG_BULL)
        caution_risk = calc.calculate(Regime.CAUTION)
        assert strong_bull_risk > caution_risk

    def test_low_win_rate_reduces_risk(self):
        calc = DynamicRiskCalculator()
        # 3 losses out of 5 = 40% win rate → should reduce
        trades = [True, False, False, False, False]
        risk = calc.calculate(Regime.BULL, recent_trades=trades)
        base = calc.calculate(Regime.BULL)
        assert risk <= base

    def test_high_win_rate_increases_risk(self):
        calc = DynamicRiskCalculator()
        trades = [True, True, True, True, True]  # 100% win rate
        risk = calc.calculate(Regime.BULL, recent_trades=trades)
        base = calc.calculate(Regime.BULL)
        assert risk >= base

    def test_risk_never_exceeds_cap(self):
        calc = DynamicRiskCalculator(max_risk_pct=0.02)
        # Even with great conditions, should cap at 2%
        trades = [True] * 10
        risk = calc.calculate(Regime.STRONG_BULL, recent_trades=trades)
        assert risk <= 0.02

    def test_risk_never_below_minimum(self):
        calc = DynamicRiskCalculator(min_risk_pct=0.005)
        trades = [False] * 10
        risk = calc.calculate(Regime.CAUTION, recent_trades=trades)
        assert risk >= 0.005


class TestATRPositionSizer:
    def test_basic_sizing(self):
        sizer = ATRPositionSizer()
        result = sizer.calculate_shares(
            capital=5000.0,
            risk_pct=0.015,
            entry_price=400.0,
            stop_price=394.0,  # $6 stop
        )
        assert result["shares"] >= 1
        # Dollar risk should be close to 1.5% of 5000 = $75
        assert result["dollar_risk"] <= 80  # approximate

    def test_larger_stop_gives_fewer_shares(self):
        sizer = ATRPositionSizer()
        result_tight = sizer.calculate_shares(5000, 0.015, 400.0, 396.0)  # $4 stop
        result_wide = sizer.calculate_shares(5000, 0.015, 400.0, 390.0)   # $10 stop
        assert result_tight["shares"] >= result_wide["shares"]

    def test_max_position_cap(self):
        sizer = ATRPositionSizer()
        result = sizer.calculate_shares(
            capital=5000.0,
            risk_pct=0.015,
            entry_price=10.0,  # cheap stock → would buy many shares
            stop_price=9.9,
            max_position_pct=0.10,
        )
        # Should not exceed 10% of capital = $500
        assert result["position_value"] <= 500.0 * 1.01  # small tolerance


class TestKillSwitch:
    def test_initially_inactive(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_file = f.name
        try:
            ks = KillSwitch(state_file=tmp_file)
            assert ks.is_active() is False
        finally:
            os.unlink(tmp_file)

    def test_activate_and_check(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_file = f.name
        try:
            ks = KillSwitch(state_file=tmp_file)
            ks.activate("Test activation")
            assert ks.is_active() is True
        finally:
            os.unlink(tmp_file)

    def test_deactivate_requires_manual_reset_by_default(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_file = f.name
        try:
            ks = KillSwitch(state_file=tmp_file)
            ks.activate("Test", requires_manual_reset=True)
            result = ks.deactivate(force=False)
            assert result is False
            assert ks.is_active() is True
        finally:
            os.unlink(tmp_file)

    def test_force_deactivate(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_file = f.name
        try:
            ks = KillSwitch(state_file=tmp_file)
            ks.activate("Test", requires_manual_reset=True)
            result = ks.deactivate(force=True)
            assert result is True
            assert ks.is_active() is False
        finally:
            os.unlink(tmp_file)

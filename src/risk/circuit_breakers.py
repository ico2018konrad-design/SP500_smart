"""Circuit Breakers — daily/weekly/monthly loss limits.

Levels:
- Per Day: -3% loss → halt until next day
- Per Week: -7% loss → halt until Monday
- Per Month: -12% drawdown → bot OFF 14 days
- Systemic: VIX > 40, SPY -5%, API errors
"""
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerStatus:
    """Current state of all circuit breakers."""
    daily_halt: bool = False
    weekly_halt: bool = False
    monthly_halt: bool = False
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    monthly_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    halt_until: Optional[datetime] = None
    halt_reason: str = ""

    def is_halted(self) -> bool:
        return self.daily_halt or self.weekly_halt or self.monthly_halt

    def to_dict(self) -> dict:
        return {
            "daily_halt": self.daily_halt,
            "weekly_halt": self.weekly_halt,
            "monthly_halt": self.monthly_halt,
            "daily_pnl_pct": self.daily_pnl_pct,
            "weekly_pnl_pct": self.weekly_pnl_pct,
            "monthly_drawdown_pct": self.monthly_drawdown_pct,
            "consecutive_losses": self.consecutive_losses,
            "halt_until": self.halt_until.isoformat() if self.halt_until else None,
            "halt_reason": self.halt_reason,
        }


class CircuitBreakers:
    """Multi-level circuit breaker system.

    Monitors P&L and halts trading when limits are breached.
    """

    def __init__(
        self,
        daily_limit: float = -0.03,
        weekly_limit: float = -0.07,
        monthly_limit: float = -0.12,
        max_consecutive_losses: int = 3,
        consecutive_cooldown_hours: int = 4,
        bot_off_days_monthly: int = 14,
        vix_flatten_threshold: float = 40.0,
        spy_panic_threshold: float = -0.05,
    ):
        self.daily_limit = daily_limit
        self.weekly_limit = weekly_limit
        self.monthly_limit = monthly_limit
        self.max_consecutive_losses = max_consecutive_losses
        self.consecutive_cooldown_hours = consecutive_cooldown_hours
        self.bot_off_days_monthly = bot_off_days_monthly
        self.vix_flatten_threshold = vix_flatten_threshold
        self.spy_panic_threshold = spy_panic_threshold

        self.status = CircuitBreakerStatus()
        self._daily_start_equity: Optional[float] = None
        self._weekly_start_equity: Optional[float] = None
        self._monthly_peak_equity: Optional[float] = None
        self._last_reset_date: Optional[date] = None

    def update(
        self,
        current_equity: float,
        closed_trade_win: Optional[bool] = None,
        vix: float = 20.0,
        spy_daily_return: float = 0.0,
    ) -> CircuitBreakerStatus:
        """Update circuit breaker state with current equity.

        Args:
            current_equity: Current portfolio equity value
            closed_trade_win: If a trade just closed: True=win, False=loss
            vix: Current VIX
            spy_daily_return: SPY return today

        Returns:
            Updated CircuitBreakerStatus
        """
        today = date.today()

        # Initialize tracking if needed — preserve any manually-set values
        if self._last_reset_date is None:
            self._last_reset_date = today
            if self._daily_start_equity is None:
                self._daily_start_equity = current_equity
            if self._weekly_start_equity is None:
                self._weekly_start_equity = current_equity
            if self._monthly_peak_equity is None:
                self._monthly_peak_equity = current_equity
        elif self._last_reset_date != today:
            # New trading day — reset daily tracking
            self._daily_start_equity = current_equity
            self.status.daily_halt = False  # Reset daily halt
            self.status.daily_pnl_pct = 0.0
            self._last_reset_date = today

        # Weekly reset (Monday)
        if today.weekday() == 0 and self._last_reset_date.weekday() != 0:
            self._weekly_start_equity = current_equity
            self.status.weekly_halt = False
            self.status.weekly_pnl_pct = 0.0

        # Update P&L tracking
        if self._daily_start_equity and self._daily_start_equity > 0:
            self.status.daily_pnl_pct = (
                (current_equity - self._daily_start_equity) / self._daily_start_equity
            )

        if self._weekly_start_equity and self._weekly_start_equity > 0:
            self.status.weekly_pnl_pct = (
                (current_equity - self._weekly_start_equity) / self._weekly_start_equity
            )

        # Update monthly peak (for drawdown)
        if self._monthly_peak_equity is None or current_equity > self._monthly_peak_equity:
            self._monthly_peak_equity = current_equity

        if self._monthly_peak_equity and self._monthly_peak_equity > 0:
            self.status.monthly_drawdown_pct = (
                (current_equity - self._monthly_peak_equity) / self._monthly_peak_equity
            )

        # Update consecutive losses
        if closed_trade_win is not None:
            if not closed_trade_win:
                self.status.consecutive_losses += 1
            else:
                self.status.consecutive_losses = 0

        # Check halt conditions (don't override existing long-term halts)
        self._check_halt_conditions(current_equity, vix, spy_daily_return)

        # Check if halt period has expired
        if self.status.halt_until and datetime.now() > self.status.halt_until:
            if self.status.daily_halt and not self.status.weekly_halt and not self.status.monthly_halt:
                self.status.daily_halt = False
                self.status.halt_until = None
                logger.info("Daily halt expired — trading resumed")

        return self.status

    def _check_halt_conditions(
        self,
        current_equity: float,
        vix: float,
        spy_daily_return: float,
    ) -> None:
        """Check all halt conditions."""
        # Daily limit: -3%
        if self.status.daily_pnl_pct <= self.daily_limit and not self.status.daily_halt:
            self.status.daily_halt = True
            self.status.halt_reason = f"Daily loss limit: {self.status.daily_pnl_pct:.1%}"
            # Find next trading day 09:00
            tomorrow = datetime.now().replace(hour=9, minute=0, second=0) + timedelta(days=1)
            self.status.halt_until = tomorrow
            logger.warning("CIRCUIT BREAKER: Daily halt activated (%.1f%% loss)", self.status.daily_pnl_pct * 100)

        # Consecutive losses: 3 → 4h cooldown
        if (self.status.consecutive_losses >= self.max_consecutive_losses and
                not self.status.daily_halt):
            self.status.daily_halt = True
            self.status.halt_until = datetime.now() + timedelta(hours=self.consecutive_cooldown_hours)
            self.status.halt_reason = f"{self.status.consecutive_losses} consecutive losses"
            logger.warning(
                "CIRCUIT BREAKER: Consecutive loss cooldown (%dh)",
                self.consecutive_cooldown_hours
            )

        # Weekly limit: -7%
        if self.status.weekly_pnl_pct <= self.weekly_limit and not self.status.weekly_halt:
            self.status.weekly_halt = True
            # Next Monday
            days_to_monday = (7 - datetime.now().weekday()) % 7
            if days_to_monday == 0:
                days_to_monday = 7
            next_monday = datetime.now() + timedelta(days=days_to_monday)
            self.status.halt_until = next_monday.replace(hour=9, minute=0, second=0)
            self.status.halt_reason = f"Weekly loss limit: {self.status.weekly_pnl_pct:.1%}"
            logger.warning("CIRCUIT BREAKER: Weekly halt activated (%.1f%% loss)", self.status.weekly_pnl_pct * 100)

        # Monthly drawdown: -12%
        if self.status.monthly_drawdown_pct <= self.monthly_limit and not self.status.monthly_halt:
            self.status.monthly_halt = True
            self.status.halt_until = datetime.now() + timedelta(days=self.bot_off_days_monthly)
            self.status.halt_reason = f"Monthly drawdown limit: {self.status.monthly_drawdown_pct:.1%}"
            logger.critical(
                "CIRCUIT BREAKER: Monthly halt activated (%.1f%% drawdown) — BOT OFF %d DAYS",
                self.status.monthly_drawdown_pct * 100,
                self.bot_off_days_monthly,
            )

    def can_trade(self) -> bool:
        """Check if trading is currently allowed."""
        return not self.status.is_halted()

    def vix_flatten_required(self, vix: float) -> bool:
        """Check if VIX requires flattening all leveraged positions."""
        return vix > self.vix_flatten_threshold

    def flash_crash_detected(self, spy_1h_return: float) -> bool:
        """Check if flash crash protection should activate."""
        return spy_1h_return <= -0.02  # SPY -2% in 1 hour

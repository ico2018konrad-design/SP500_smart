"""Reactive Hedge Module — dynamic triggers.

Triggers:
1. SPY -1% intraday AND VIX +10%: buy SPXS 10% capital
2. Regime downgrade BULL→CHOP: increase SH to 20%
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ReactiveHedgeAction:
    """Action triggered by reactive hedge."""
    action: str         # "buy_spxs", "increase_sh", "none"
    instrument: str
    capital_pct: float  # % of capital to allocate
    reason: str
    timestamp: datetime
    stop_condition: str   # when to close reactive hedge
    close_condition: str  # target close


class ReactiveHedge:
    """Dynamic hedge that activates on market stress signals.

    Level 2 of hedge system — sits above always-on baseline.
    """

    def __init__(
        self,
        spy_drop_threshold: float = -0.01,   # SPY -1% intraday
        vix_spike_threshold: float = 0.10,   # VIX +10%
        reactive_hedge_pct: float = 0.10,    # SPXS 10% capital
        sh_increase_pct: float = 0.20,       # SH 20% on downgrade
    ):
        self.spy_drop_threshold = spy_drop_threshold
        self.vix_spike_threshold = vix_spike_threshold
        self.reactive_hedge_pct = reactive_hedge_pct
        self.sh_increase_pct = sh_increase_pct
        self._reactive_hedge_active = False
        self._reactive_entry_spy = None

    def check_triggers(
        self,
        spy_intraday_return: float,
        vix_intraday_change: float,
        current_regime: str,
        previous_regime: str,
    ) -> Optional[ReactiveHedgeAction]:
        """Check if reactive hedge should activate.

        Args:
            spy_intraday_return: SPY intraday return (negative = down)
            vix_intraday_change: VIX change % today
            current_regime: Current regime string
            previous_regime: Previous regime string

        Returns:
            ReactiveHedgeAction if triggered, None otherwise
        """
        # Trigger 1: SPY -1% AND VIX +10% simultaneously
        if (spy_intraday_return <= self.spy_drop_threshold and
                vix_intraday_change >= self.vix_spike_threshold):
            self._reactive_hedge_active = True
            logger.warning(
                "Reactive hedge TRIGGER: SPY %.1f%%, VIX +%.1f%%",
                spy_intraday_return * 100, vix_intraday_change * 100
            )
            return ReactiveHedgeAction(
                action="buy_spxs",
                instrument="SPXS",
                capital_pct=self.reactive_hedge_pct,
                reason=f"SPY {spy_intraday_return:.1%} + VIX {vix_intraday_change:.1%}",
                timestamp=datetime.now(),
                stop_condition="Close when SPY +0.5% off intraday low",
                close_condition="Close when SPY +1% off intraday low",
            )

        # Trigger 2: Regime downgrade BULL→CHOP within 1 day
        if previous_regime == "BULL" and current_regime == "CHOP":
            logger.warning("Reactive hedge TRIGGER: Regime downgrade BULL→CHOP")
            return ReactiveHedgeAction(
                action="increase_sh",
                instrument="SH",
                capital_pct=self.sh_increase_pct,  # increase to 20%
                reason="Regime downgrade BULL→CHOP",
                timestamp=datetime.now(),
                stop_condition="Regime upgrades back to BULL",
                close_condition="Regime returns to BULL or higher",
            )

        return None

    def check_close_conditions(
        self,
        spy_current: float,
        spy_intraday_low: float,
    ) -> bool:
        """Check if reactive SPXS hedge should be closed.

        Close when SPY recovers +1% from intraday low.
        """
        if not self._reactive_hedge_active or spy_intraday_low is None:
            return False

        recovery = (spy_current - spy_intraday_low) / spy_intraday_low
        should_close = recovery >= 0.01  # +1% recovery

        if should_close:
            self._reactive_hedge_active = False
            logger.info(
                "Reactive hedge CLOSED: SPY recovered %.1f%% from intraday low",
                recovery * 100
            )

        return should_close

    @property
    def is_active(self) -> bool:
        return self._reactive_hedge_active

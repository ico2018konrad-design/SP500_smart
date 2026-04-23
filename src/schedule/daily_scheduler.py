"""Daily Scheduler — orchestrates the daily trading routine.

Schedule (CET):
07:00 Morning briefing
15:30 Open blackout start
16:00 Active trading start
21:30 Pre-close blackout
22:00 Daily reconciliation
22:00-07:00 Sleep mode
"""
import logging
import os
import time
from datetime import datetime
from typing import Optional

import schedule

logger = logging.getLogger(__name__)


class DailyScheduler:
    """Orchestrates all daily bot routines."""

    def __init__(self, paper_trader=None, regime_detector=None):
        self.paper_trader = paper_trader
        self.regime_detector = regime_detector
        self._running = False
        self._in_blackout = False

    def morning_briefing(self) -> None:
        """07:00 CET — Morning analysis and regime recalculation."""
        logger.info("=" * 50)
        logger.info("MORNING BRIEFING — %s", datetime.now().strftime("%Y-%m-%d %H:%M"))

        try:
            from src.data.yahoo_loader import load_spy, load_vix
            from src.data.fred_macro import get_current_hy_spread, get_current_yield_curve

            spy = load_spy()
            vix = load_vix()

            if not spy.empty and not vix.empty:
                from src.regime.detector import RegimeDetector
                detector = RegimeDetector()
                hy_spread = get_current_hy_spread()
                yc = get_current_yield_curve()
                result = detector.detect(
                    spy_daily=spy,
                    vix_daily=vix,
                    breadth_pct=0.55,
                    hy_spread_bps=hy_spread,
                    yield_curve_freshly_inverted=(yc < 0),
                )
                logger.info(
                    "Regime: %s (score=%d/11) | VIX: %.1f | HY: %.0fbps",
                    result.regime.value, result.score,
                    float(vix["Close"].iloc[-1]) if not vix.empty else 0,
                    hy_spread,
                )
        except Exception as exc:
            logger.error("Morning briefing error: %s", exc)

    def open_blackout_start(self) -> None:
        """15:30 CET — US open blackout begins."""
        self._in_blackout = True
        logger.info("BLACKOUT: US Open blackout started (15:30 CET)")

    def trading_start(self) -> None:
        """16:00 CET — Active trading window opens."""
        self._in_blackout = False
        logger.info("TRADING: Active trading window open (16:00 CET)")

    def pre_close_blackout(self) -> None:
        """21:30 CET — Pre-close blackout."""
        self._in_blackout = True
        logger.info("BLACKOUT: Pre-close blackout (21:30 CET)")

    def daily_reconciliation(self) -> None:
        """22:00 CET — Daily reconciliation and reports."""
        self._in_blackout = False
        logger.info("RECONCILIATION: Daily close — %s", datetime.now().strftime("%Y-%m-%d"))

        if self.paper_trader:
            equity = self.paper_trader.get_equity()
            positions = self.paper_trader.position_manager.get_position_count()
            logger.info("EOD Equity: $%.2f | Open Positions: %d", equity, positions)

    def setup_schedule(self) -> None:
        """Set up the daily schedule."""
        schedule.every().day.at("07:00").do(self.morning_briefing)
        schedule.every().day.at("15:30").do(self.open_blackout_start)
        schedule.every().day.at("16:00").do(self.trading_start)
        schedule.every().day.at("21:30").do(self.pre_close_blackout)
        schedule.every().day.at("22:00").do(self.daily_reconciliation)

        logger.info("Daily schedule configured")

    def run(self) -> None:
        """Run the scheduler indefinitely."""
        self.setup_schedule()
        self._running = True
        logger.info("Scheduler started")

        while self._running:
            schedule.run_pending()
            time.sleep(30)

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        logger.info("Scheduler stopped")

    @property
    def is_blackout(self) -> bool:
        return self._in_blackout

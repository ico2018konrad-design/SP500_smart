"""Daily Report — email/log summary at 22:00."""
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class DailyReport:
    """Generates and sends daily trading report."""

    def generate(
        self,
        equity: float,
        starting_equity: float,
        daily_pnl_pct: float,
        weekly_pnl_pct: float,
        monthly_drawdown_pct: float,
        regime: str,
        regime_score: int,
        open_positions: int,
        trades_today: int,
        vix: float,
    ) -> str:
        """Generate daily report text."""
        total_return = (equity / starting_equity - 1) if starting_equity > 0 else 0

        report = f"""
╔══════════════════════════════════════════════════╗
║         SP500 Smart Bot — Daily Report           ║
║         {datetime.now().strftime("%Y-%m-%d %H:%M CET")}                 ║
╠══════════════════════════════════════════════════╣
║ EQUITY & PERFORMANCE                              ║
║  Current Equity:     ${equity:>10,.2f}             ║
║  Daily P&L:          {daily_pnl_pct:>+10.2%}             ║
║  Weekly P&L:         {weekly_pnl_pct:>+10.2%}             ║
║  Monthly DD:         {monthly_drawdown_pct:>+10.2%}             ║
║  Total Return:       {total_return:>+10.2%}             ║
╠══════════════════════════════════════════════════╣
║ MARKET REGIME                                     ║
║  Regime:             {regime:<20}        ║
║  Score:              {regime_score}/11                          ║
║  VIX:                {vix:<20.1f}        ║
╠══════════════════════════════════════════════════╣
║ ACTIVITY                                          ║
║  Open Positions:     {open_positions:<20}        ║
║  Trades Today:       {trades_today:<20}        ║
╚══════════════════════════════════════════════════╝
"""
        logger.info(report)
        return report

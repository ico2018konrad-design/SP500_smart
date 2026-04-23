"""Baseline Hedge Module — always-active insurance.

Phase 1 (capital < 15k): 10% capital in SH (-1x inverse)
Phase 2 (capital >= 15k): SPY put spreads (full mode only)
"""
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HedgeStatus:
    """Current state of baseline hedge."""
    mode: str              # "mini" or "full"
    capital: float
    hedge_instrument: str  # SH or SPY_PUT_SPREAD
    hedge_pct: float       # % of capital in hedge
    hedge_value: float     # dollar value in hedge
    is_active: bool
    details: dict


class BaselineHedge:
    """Manages the always-active baseline hedge position.

    Mini mode: Keep 10% of capital in SH at all times.
    Full mode: SPY put spreads for downside protection.
    """

    MINI_INSTRUMENT = "SH"
    FULL_INSTRUMENT = "SPY_PUT_SPREAD"
    FULL_MODE_THRESHOLD = 15000.0

    def __init__(
        self,
        mode: str = "mini",
        mini_hedge_pct: float = 0.10,
        full_put_spread_pct: float = 0.06,
        put_spread_dte: int = 90,
        put_long_moneyness: float = 0.93,
        put_short_moneyness: float = 0.82,
    ):
        self.mode = mode
        self.mini_hedge_pct = mini_hedge_pct
        self.full_put_spread_pct = full_put_spread_pct
        self.put_spread_dte = put_spread_dte
        self.put_long_moneyness = put_long_moneyness
        self.put_short_moneyness = put_short_moneyness

    def get_hedge_allocation(self, capital: float) -> HedgeStatus:
        """Calculate hedge allocation based on current capital.

        Returns HedgeStatus with hedge details.
        """
        # Determine if we should use full mode (capital >= 15k)
        use_full_mode = self.mode == "full" and capital >= self.FULL_MODE_THRESHOLD

        if use_full_mode:
            hedge_pct = self.full_put_spread_pct
            instrument = self.FULL_INSTRUMENT
            details = {
                "type": "put_spread",
                "long_put_moneyness": self.put_long_moneyness,
                "short_put_moneyness": self.put_short_moneyness,
                "dte": self.put_spread_dte,
                "monthly_cost_pct": self.full_put_spread_pct,
            }
        else:
            hedge_pct = self.mini_hedge_pct
            instrument = self.MINI_INSTRUMENT
            details = {
                "type": "inverse_etf",
                "instrument": self.MINI_INSTRUMENT,
                "shares_pct": self.mini_hedge_pct,
            }

        hedge_value = capital * hedge_pct

        logger.debug(
            "Baseline hedge: %s %.1f%% = $%.2f",
            instrument, hedge_pct * 100, hedge_value
        )

        return HedgeStatus(
            mode="full" if use_full_mode else "mini",
            capital=capital,
            hedge_instrument=instrument,
            hedge_pct=hedge_pct,
            hedge_value=hedge_value,
            is_active=True,
            details=details,
        )

    def get_sh_shares(self, capital: float, sh_price: float) -> int:
        """Calculate number of SH shares for mini mode hedge.

        Args:
            capital: Available capital
            sh_price: Current SH price

        Returns:
            Number of SH shares to hold
        """
        hedge_value = capital * self.mini_hedge_pct
        shares = max(0, int(hedge_value / sh_price)) if sh_price > 0 else 0
        return shares

    def needs_rebalancing(
        self,
        current_hedge_pct: float,
        target_hedge_pct: float,
        tolerance: float = 0.02,
    ) -> bool:
        """Check if hedge needs rebalancing.

        Rebalance if deviation > tolerance (default 2%).
        """
        return abs(current_hedge_pct - target_hedge_pct) > tolerance

    def get_put_spread_strikes(self, spy_price: float) -> dict:
        """Calculate put spread strikes for full mode.

        Long put @ 93% of SPY, short put @ 82% of SPY.
        """
        long_put_strike = round(spy_price * self.put_long_moneyness, 0)
        short_put_strike = round(spy_price * self.put_short_moneyness, 0)

        return {
            "spy_price": spy_price,
            "long_put_strike": long_put_strike,  # protective put
            "short_put_strike": short_put_strike,  # sold put (offset cost)
            "max_protection_pct": 1 - self.put_short_moneyness,  # 18% protection
            "cost_pct_of_spy": self.full_put_spread_pct,
        }

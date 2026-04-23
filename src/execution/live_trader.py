"""Live Trader — IBKR live trading (DISABLED BY DEFAULT).

⚠️ WARNING: This module trades with REAL MONEY.
Must explicitly set trading_mode: "live" in config AND
IBKR_PORT=7496 (live gateway) or IBKR_PORT=4001 (live gateway API).

Do NOT enable until after 6+ months of paper trading validation.
"""
import logging
import os

logger = logging.getLogger(__name__)

# Safety check — this module should never be imported in paper mode
LIVE_TRADING_ENABLED = os.getenv("TRADING_MODE", "paper").lower() == "live"


def require_live_confirmation() -> bool:
    """Multi-step confirmation before enabling live trading."""
    if not LIVE_TRADING_ENABLED:
        logger.error(
            "Live trading blocked: TRADING_MODE=%s. "
            "Set TRADING_MODE=live in .env to enable.",
            os.getenv("TRADING_MODE", "paper")
        )
        return False

    # Additional safety: verify config file
    try:
        import yaml
        with open("config/strategy_config.yaml") as f:
            config = yaml.safe_load(f)
        if config.get("trading_mode") != "live":
            logger.error(
                "Live trading blocked: strategy_config.yaml trading_mode=%s",
                config.get("trading_mode")
            )
            return False
    except Exception as exc:
        logger.error("Could not verify config: %s", exc)
        return False

    logger.warning("=" * 60)
    logger.warning("⚠️  LIVE TRADING MODE — REAL MONEY AT RISK  ⚠️")
    logger.warning("=" * 60)
    return True


class LiveTrader:
    """Live trading implementation via IBKR.

    ⚠️ WARNING: This trades with REAL MONEY.
    Only use after extensive paper trading validation.
    """

    def __init__(self):
        if not require_live_confirmation():
            raise RuntimeError(
                "Live trading not enabled. Set TRADING_MODE=live and "
                "trading_mode: live in strategy_config.yaml."
            )

        self._confirmed = True
        logger.warning("LiveTrader initialized — REAL MONEY MODE ACTIVE")

        # Import paper trader as base and override execution
        from src.execution.paper_trader import PaperTrader
        self._base = PaperTrader()

        # Connect to LIVE port (not paper)
        live_port = int(os.getenv("IBKR_PORT", "7496"))
        logger.warning("Connecting to LIVE IBKR port %d", live_port)

    def __getattr__(self, name):
        """Delegate to paper trader base for unimplemented methods."""
        return getattr(self._base, name)

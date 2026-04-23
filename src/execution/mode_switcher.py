"""Mode Switcher — mini vs full mode."""
import logging
import os

import yaml

logger = logging.getLogger(__name__)


def get_current_mode() -> str:
    """Get current bot mode from config."""
    try:
        with open("config/strategy_config.yaml") as f:
            config = yaml.safe_load(f)
        return config.get("mode", "mini")
    except Exception:
        return "mini"


def get_trading_mode() -> str:
    """Get current trading mode (paper/live)."""
    env_mode = os.getenv("TRADING_MODE", "").lower()
    if env_mode in ("live", "paper"):
        return env_mode
    try:
        with open("config/strategy_config.yaml") as f:
            config = yaml.safe_load(f)
        return config.get("trading_mode", "paper")
    except Exception:
        return "paper"


def is_mini_mode() -> bool:
    return get_current_mode() == "mini"


def is_full_mode() -> bool:
    return get_current_mode() == "full"


def is_paper_trading() -> bool:
    return get_trading_mode() == "paper"


def is_live_trading() -> bool:
    return get_trading_mode() == "live"


def get_allowed_instruments() -> list:
    """Get list of allowed instruments for current mode."""
    try:
        with open("config/strategy_config.yaml") as f:
            config = yaml.safe_load(f)
        mode = config.get("mode", "mini")
        instruments = config.get("instruments", {})
        return instruments.get(mode, ["SPY", "UPRO", "SH", "SPXS"])
    except Exception:
        return ["SPY", "UPRO", "SH", "SPXS"]

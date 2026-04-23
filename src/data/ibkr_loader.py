"""Interactive Brokers real-time data loader via ib_insync.

Connects to IBKR TWS or Gateway.
Default: paper trading on port 7497.
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", "7497"))  # 7497=paper, 4002=live gateway
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "1"))


def _get_ib():
    """Create IB connection. Returns None if ib_insync unavailable."""
    try:
        from ib_insync import IB
        return IB()
    except ImportError:
        logger.error("ib_insync not installed. Run: pip install ib_insync")
        return None


def get_ibkr_contract(symbol: str, sec_type: str = "STK",
                      exchange: str = "SMART", currency: str = "USD"):
    """Create IBKR contract object for a symbol."""
    try:
        from ib_insync import Stock
        return Stock(symbol, exchange, currency)
    except ImportError:
        return None


class IBKRLoader:
    """Interactive Brokers data loader.

    Usage:
        loader = IBKRLoader()
        loader.connect()
        data = loader.get_historical_data("SPY", "1 D", "5 mins")
        loader.disconnect()
    """

    def __init__(self):
        self.ib = _get_ib()
        self.connected = False

    def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway."""
        if self.ib is None:
            logger.error("ib_insync not available")
            return False

        try:
            self.ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)
            self.connected = True
            logger.info(
                "Connected to IBKR at %s:%d (client_id=%d)",
                IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID
            )
            return True
        except Exception as exc:
            logger.error("Failed to connect to IBKR: %s", exc)
            logger.error(
                "Make sure TWS/Gateway is running. "
                "Paper trading port: 7497. Live port: 7496/4001."
            )
            return False

    def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")

    def get_historical_data(
        self,
        symbol: str,
        duration: str = "5 D",
        bar_size: str = "5 mins",
        what_to_show: str = "TRADES",
    ) -> pd.DataFrame:
        """Get historical OHLCV data from IBKR.

        Args:
            symbol: Ticker symbol
            duration: Duration string (e.g., '5 D', '1 M', '1 Y')
            bar_size: Bar size (e.g., '5 mins', '1 hour', '1 day')
            what_to_show: Data type (TRADES, BID, ASK, MIDPOINT)

        Returns:
            DataFrame with OHLCV data.
        """
        if not self.connected:
            logger.warning("Not connected to IBKR. Call connect() first.")
            return pd.DataFrame()

        try:
            contract = get_ibkr_contract(symbol)
            if contract is None:
                return pd.DataFrame()

            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,
                formatDate=1,
            )

            if not bars:
                logger.warning("No historical data returned for %s", symbol)
                return pd.DataFrame()

            from ib_insync import util
            df = util.df(bars)
            df.index = pd.to_datetime(df["date"])
            df = df.drop(columns=["date"], errors="ignore")
            df.columns = [c.capitalize() for c in df.columns]
            logger.info("Loaded %d IBKR bars for %s", len(df), symbol)
            return df

        except Exception as exc:
            logger.error("Failed to get historical data for %s: %s", symbol, exc)
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current last price for a symbol."""
        if not self.connected:
            return None

        try:
            contract = get_ibkr_contract(symbol)
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract, "", False, False)
            self.ib.sleep(1.0)

            price = ticker.last or ticker.close
            if price and price > 0:
                return float(price)
            return None

        except Exception as exc:
            logger.error("Failed to get current price for %s: %s", symbol, exc)
            return None

    def get_account_summary(self) -> dict:
        """Get account summary (equity, cash, etc.)."""
        if not self.connected:
            return {}

        try:
            summary = self.ib.accountSummary()
            result = {}
            for item in summary:
                result[item.tag] = item.value
            return result
        except Exception as exc:
            logger.error("Failed to get account summary: %s", exc)
            return {}

    def get_positions(self) -> list:
        """Get current open positions."""
        if not self.connected:
            return []

        try:
            positions = self.ib.positions()
            result = []
            for pos in positions:
                result.append({
                    "symbol": pos.contract.symbol,
                    "shares": pos.position,
                    "avg_cost": pos.avgCost,
                    "market_value": pos.marketValue if hasattr(pos, "marketValue") else None,
                })
            return result
        except Exception as exc:
            logger.error("Failed to get positions: %s", exc)
            return []

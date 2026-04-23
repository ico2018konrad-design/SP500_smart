"""Paper Trader — IBKR paper trading mode.

Connects to IBKR paper account (port 7497).
Simulates execution with realistic slippage and commission.
"""
import logging
import os
from datetime import datetime
from typing import Dict, Optional

from src.execution.order_manager import Order, OrderManager
from src.positions.position_manager import Position, PositionManager
from src.signals.signal_types import Signal, SignalDirection
from src.risk.atr_position_sizer import ATRPositionSizer

logger = logging.getLogger(__name__)

COMMISSION_PCT = 0.0005   # 0.05% per trade (IBKR)
SLIPPAGE_PCT = 0.001      # 0.1% slippage estimate


class PaperTrader:
    """Paper trading implementation.

    Executes trades against IBKR paper account.
    Falls back to local simulation if IBKR unavailable.
    """

    def __init__(
        self,
        starting_capital: float = 5000.0,
        commission_pct: float = COMMISSION_PCT,
        slippage_pct: float = SLIPPAGE_PCT,
    ):
        self.starting_capital = starting_capital
        self.capital = starting_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.order_manager = OrderManager()
        self.position_manager = PositionManager()
        self.sizer = ATRPositionSizer()
        self._ibkr_connected = False
        self._ibkr_loader = None
        self._trade_log = []

    def connect_ibkr(self) -> bool:
        """Try to connect to IBKR paper account."""
        try:
            from src.data.ibkr_loader import IBKRLoader
            self._ibkr_loader = IBKRLoader()
            self._ibkr_connected = self._ibkr_loader.connect()
            if self._ibkr_connected:
                logger.info("Connected to IBKR paper account")
            else:
                logger.warning("IBKR not available — using local simulation")
            return self._ibkr_connected
        except Exception as exc:
            logger.warning("IBKR connection failed: %s — using simulation", exc)
            return False

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from IBKR or fallback to Yahoo."""
        if self._ibkr_connected and self._ibkr_loader:
            price = self._ibkr_loader.get_current_price(symbol)
            if price:
                return price

        # Fallback: Yahoo Finance
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass

        return None

    def execute_signal(
        self,
        signal: Signal,
        risk_pct: float = 0.015,
        allocated_capital: Optional[float] = None,
    ) -> Optional[Position]:
        """Execute a trading signal.

        Args:
            signal: Trading signal to execute
            risk_pct: Risk per trade as decimal
            allocated_capital: Capital allocated for this trade

        Returns:
            Position if executed, None if failed
        """
        if allocated_capital is None:
            allocated_capital = self.capital

        symbol = signal.symbol
        direction = signal.direction.value

        # Get current price (with slippage)
        current_price = self.get_current_price(symbol)
        if current_price is None:
            current_price = signal.entry_price

        # Apply slippage
        if direction == "LONG":
            fill_price = current_price * (1 + self.slippage_pct)
        else:
            fill_price = current_price * (1 - self.slippage_pct)

        # Calculate position size
        sizing = self.sizer.calculate_shares(
            capital=allocated_capital,
            risk_pct=risk_pct,
            entry_price=fill_price,
            stop_price=signal.stop_price,
        )
        shares = sizing["shares"]

        if shares <= 0:
            logger.warning("Position size is 0 for %s — insufficient capital?", symbol)
            return None

        # Debit full position cost + commission from capital on open
        commission = shares * fill_price * self.commission_pct
        self.capital -= shares * fill_price + commission

        # Create and fill order
        action = "BUY" if direction == "LONG" else "SELL"
        order = self.order_manager.create_market_order(symbol, action, shares)
        self.order_manager.simulate_fill(order.order_id, fill_price)

        # Open position
        position = self.position_manager.add_position(
            symbol=symbol,
            direction=direction,
            entry_price=fill_price,
            shares=shares,
            stop_price=signal.stop_price,
            target1=signal.target1,
            target2=signal.target2,
            target3=signal.target3,
        )

        if position:
            self._log_trade(
                "OPEN", symbol, direction, shares, fill_price,
                signal.regime, signal.regime_score
            )

        return position

    def close_position(
        self,
        position_id: str,
        reason: str = "manual",
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Optional[float]:
        """Close a position.

        Returns realized PnL % if successful.
        """
        pos = self.position_manager.positions.get(position_id)
        if pos is None:
            return None

        current_price = None
        if current_prices:
            current_price = current_prices.get(pos.symbol)
        if current_price is None:
            current_price = self.get_current_price(pos.symbol)
        if current_price is None:
            current_price = pos.current_price

        # Apply slippage (unfavorable direction)
        if pos.direction == "LONG":
            fill_price = current_price * (1 - self.slippage_pct)
        else:
            fill_price = current_price * (1 + self.slippage_pct)

        shares_open = pos.shares_open

        pnl_pct = self.position_manager.close_position(position_id, fill_price, reason)

        if pnl_pct is not None:
            # Credit the sale proceeds minus commission (cost was debited on open)
            commission = shares_open * fill_price * self.commission_pct
            self.capital += shares_open * fill_price - commission
            self._log_trade("CLOSE", pos.symbol, pos.direction, pos.shares, fill_price, reason=reason, pnl_pct=pnl_pct)

        return pnl_pct

    def get_equity(self) -> float:
        """Get current portfolio equity (cash + mark-to-market open positions)."""
        equity = self.capital
        for pos in self.position_manager.get_open_positions():
            current_price = self.get_current_price(pos.symbol) or pos.current_price
            if pos.direction == "LONG":
                equity += pos.shares_open * current_price
            else:
                # For short positions: we received proceeds at entry, now owe current_price
                equity -= pos.shares_open * current_price
        return equity

    def _log_trade(
        self,
        action: str,
        symbol: str,
        direction: str,
        shares: int,
        price: float,
        regime: str = "",
        regime_score: int = 0,
        reason: str = "",
        pnl_pct: Optional[float] = None,
    ) -> None:
        """Log trade to internal log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "symbol": symbol,
            "direction": direction,
            "shares": shares,
            "price": price,
            "regime": regime,
            "regime_score": regime_score,
            "reason": reason,
            "pnl_pct": pnl_pct,
            "capital": self.capital,
        }
        self._trade_log.append(entry)
        logger.info(
            "Trade: %s %s %s %d @ %.2f%s",
            action, direction, symbol, shares, price,
            f" | PnL: {pnl_pct:.2%}" if pnl_pct is not None else ""
        )

    def get_trade_log(self) -> list:
        return self._trade_log.copy()

"""Trade Journal — logs every trade with reason."""
import csv
import json
import logging
import os
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


class TradeJournal:
    """Logs all trades with full details for analysis."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_file = os.path.join(log_dir, "trade_journal.csv")
        self.json_file = os.path.join(log_dir, "trade_journal.json")
        self._trades: List[dict] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file) as f:
                    self._trades = json.load(f)
            except Exception:
                self._trades = []

    def log_trade(
        self,
        action: str,
        symbol: str,
        direction: str,
        shares: int,
        price: float,
        regime: str = "",
        regime_score: int = 0,
        signal_triggers: Optional[List[str]] = None,
        signal_confirms: Optional[List[str]] = None,
        stop_price: float = 0.0,
        target1: float = 0.0,
        reason: str = "",
        pnl_pct: Optional[float] = None,
        notes: str = "",
    ) -> None:
        """Log a trade entry or exit."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "symbol": symbol,
            "direction": direction,
            "shares": shares,
            "price": price,
            "regime": regime,
            "regime_score": regime_score,
            "signal_triggers": signal_triggers or [],
            "signal_confirms": signal_confirms or [],
            "stop_price": stop_price,
            "target1": target1,
            "reason": reason,
            "pnl_pct": pnl_pct,
            "notes": notes,
        }
        self._trades.append(entry)
        self._save()
        logger.info(
            "Journal: %s %s %s %d @ %.2f%s",
            action, direction, symbol, shares, price,
            f" PnL:{pnl_pct:.2%}" if pnl_pct is not None else ""
        )

    def _save(self) -> None:
        try:
            with open(self.json_file, "w") as f:
                json.dump(self._trades, f, indent=2, default=str)
        except Exception as exc:
            logger.error("Failed to save trade journal: %s", exc)

    def get_recent_win_rate(self, n: int = 10) -> float:
        closed = [t for t in self._trades if t.get("action") == "CLOSE"]
        recent = closed[-n:]
        if not recent:
            return 0.5
        wins = sum(1 for t in recent if (t.get("pnl_pct") or 0) > 0)
        return wins / len(recent)

    def get_all_trades(self) -> List[dict]:
        return self._trades.copy()

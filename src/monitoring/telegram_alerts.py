"""Telegram alerts — entry/exit, warnings, panic alerts."""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class TelegramAlerter:
    """Sends trading alerts via Telegram bot."""

    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self._bot = None

    def _get_bot(self):
        if not self.token:
            return None
        try:
            from telegram import Bot
            if self._bot is None:
                self._bot = Bot(token=self.token)
            return self._bot
        except ImportError:
            logger.warning("python-telegram-bot not installed")
            return None

    def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send a message to the Telegram chat."""
        bot = self._get_bot()
        if bot is None or not self.chat_id:
            logger.debug("Telegram not configured: %s", message)
            return False
        try:
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                bot.send_message(chat_id=self.chat_id, text=message, parse_mode=parse_mode)
            )
            return True
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False

    def trade_entry(self, symbol: str, direction: str, price: float,
                    shares: int, stop: float, regime: str) -> None:
        msg = (
            f"📈 <b>TRADE ENTRY</b>\n"
            f"Symbol: {symbol} | {direction}\n"
            f"Price: ${price:.2f} | Shares: {shares}\n"
            f"Stop: ${stop:.2f}\n"
            f"Regime: {regime}"
        )
        self.send(msg)

    def trade_exit(self, symbol: str, pnl_pct: float, reason: str) -> None:
        emoji = "✅" if pnl_pct > 0 else "❌"
        msg = (
            f"{emoji} <b>TRADE EXIT</b>\n"
            f"Symbol: {symbol}\n"
            f"PnL: {pnl_pct:+.2%}\n"
            f"Reason: {reason}"
        )
        self.send(msg)

    def panic_alert(self, reason: str, vix: float) -> None:
        msg = (
            f"🚨 <b>PANIC MODE ACTIVATED</b> 🚨\n"
            f"Reason: {reason}\n"
            f"VIX: {vix:.1f}\n"
            f"Action: All longs closed. SPXS+VXX activated.\n"
            f"⚠️ MANUAL UNLOCK REQUIRED to resume"
        )
        self.send(msg)

    def circuit_breaker_alert(self, reason: str, halt_until: str) -> None:
        msg = (
            f"⛔ <b>CIRCUIT BREAKER</b>\n"
            f"Reason: {reason}\n"
            f"Trading halted until: {halt_until}"
        )
        self.send(msg)

    def daily_summary(self, equity: float, daily_pnl_pct: float,
                      regime: str, open_positions: int) -> None:
        emoji = "📈" if daily_pnl_pct > 0 else "📉"
        msg = (
            f"{emoji} <b>DAILY SUMMARY</b>\n"
            f"Equity: ${equity:,.2f}\n"
            f"Day PnL: {daily_pnl_pct:+.2%}\n"
            f"Regime: {regime}\n"
            f"Open Positions: {open_positions}"
        )
        self.send(msg)

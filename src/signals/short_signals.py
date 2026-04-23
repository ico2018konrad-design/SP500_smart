"""Short signal generation — 3-level check (Setup, Trigger, Confirmation)."""
import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd

from src.signals.signal_types import Signal, SignalDirection, SignalStrength
from src.signals.indicators import (
    calc_rsi, calc_macd, calc_stochastic, calc_bollinger_bands,
    calc_atr_pct, calc_ema,
    is_rsi_crossing_down, is_price_touching_ema, is_macd_hist_rising,
    is_volume_elevated,
)

logger = logging.getLogger(__name__)


class ShortSignalGenerator:
    """Generates SHORT entry signals (via SPXS or SH).

    Level 1 — SETUP (ALL required):
        - Regime score <= 5
        - VIX > 20 and rising
        - SPY < 50 SMA
        - No panic rebound in progress

    Level 2 — TRIGGER (min 3 of 5):
        1. RSI > 68 crossing down
        2. Price rejected from 50 EMA
        3. MACD bearish cross
        4. Stochastic > 75, bearish cross
        5. BB upper rejection

    Level 3 — CONFIRMATION (min 2 of 4):
        1. Red volume > green (3-bar)
        2. VIX rising > 5% intraday
        3. Put/Call > 1.0
        4. A/D falling

    Execute SHORT via SPXS (-3x) or SH (-1x), NOT naked short.
    Size 1/3 of normal long.
    """

    def __init__(
        self,
        max_regime_score: int = 5,
        vix_threshold: float = 20.0,
        min_triggers: int = 3,
        min_confirms: int = 2,
        stop_pct: float = 0.015,
        target1_pct: float = 0.020,
        target2_pct: float = 0.040,
        min_rr: float = 1.5,
        size_multiplier: float = 0.333,  # 1/3 of normal
    ):
        self.max_regime_score = max_regime_score
        self.vix_threshold = vix_threshold
        self.min_triggers = min_triggers
        self.min_confirms = min_confirms
        self.stop_pct = stop_pct
        self.target1_pct = target1_pct
        self.target2_pct = target2_pct
        self.min_rr = min_rr
        self.size_multiplier = size_multiplier

    def check_setup(
        self,
        regime_score: int,
        vix: float,
        vix_rising: bool,
        spy_below_50sma: bool,
        panic_rebound: bool = False,
    ) -> tuple:
        """Level 1: Check all setup conditions."""
        reasons = []

        if regime_score > self.max_regime_score:
            reasons.append(f"regime_score {regime_score} > max {self.max_regime_score}")

        if not (vix > self.vix_threshold and vix_rising):
            reasons.append(f"VIX {vix:.1f} not above {self.vix_threshold} or not rising")

        if not spy_below_50sma:
            reasons.append("SPY above 50 SMA (need below for short)")

        if panic_rebound:
            reasons.append("panic rebound in progress — avoid shorting")

        return len(reasons) == 0, reasons

    def check_triggers(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        open_: Optional[pd.Series] = None,
    ) -> tuple:
        """Level 2: Check trigger conditions (min 3 of 5)."""
        triggers_hit = []

        rsi = calc_rsi(close, 14)
        ema50 = calc_ema(close, 50)
        macd_line, signal_line, macd_hist = calc_macd(close)
        stoch_k, stoch_d = calc_stochastic(high, low, close)
        bb_upper, bb_mid, bb_lower = calc_bollinger_bands(close)

        # Trigger 1: RSI > 68, crossing down
        if len(rsi) > 3:
            current_rsi = float(rsi.dropna().iloc[-1])
            if current_rsi > 68 and is_rsi_crossing_down(rsi, 68):
                triggers_hit.append("RSI_crossdown_68")
            elif current_rsi > 68:
                triggers_hit.append("RSI_overbought_68")

        # Trigger 2: Price rejected from 50 EMA (price < EMA after touching it)
        if len(ema50) > 3:
            curr_close = float(close.dropna().iloc[-1])
            curr_ema = float(ema50.dropna().iloc[-1])
            if is_price_touching_ema(close, ema50) and curr_close < curr_ema:
                triggers_hit.append("EMA50_rejection")

        # Trigger 3: MACD bearish cross (MACD crosses below Signal)
        if len(macd_line) > 3 and len(signal_line) > 3:
            ml = macd_line.dropna()
            sl = signal_line.dropna()
            if len(ml) >= 2 and len(sl) >= 2:
                prev_above = float(ml.iloc[-2]) >= float(sl.iloc[-2])
                curr_below = float(ml.iloc[-1]) < float(sl.iloc[-1])
                if prev_above and curr_below:
                    triggers_hit.append("MACD_bearish_cross")

        # Trigger 4: Stochastic > 75, bearish cross (%K crosses below %D)
        if len(stoch_k) > 3 and len(stoch_d) > 3:
            curr_k = float(stoch_k.dropna().iloc[-1])
            prev_k = float(stoch_k.dropna().iloc[-2]) if len(stoch_k.dropna()) > 1 else curr_k
            curr_d = float(stoch_d.dropna().iloc[-1])
            prev_d = float(stoch_d.dropna().iloc[-2]) if len(stoch_d.dropna()) > 1 else curr_d
            if curr_k > 75 and prev_k >= prev_d and curr_k < curr_d:
                triggers_hit.append("Stoch_bearish_cross_75")
            elif curr_k > 75:
                triggers_hit.append("Stoch_overbought_75")

        # Trigger 5: BB upper rejection (price touched upper then closed below)
        if len(bb_upper) > 1 and len(close) > 1:
            recent_close = close.dropna().tail(3)
            recent_upper = bb_upper.dropna().tail(3)
            touched_upper = any(c >= u * 0.998 for c, u in zip(recent_close, recent_upper))
            curr_below = float(close.dropna().iloc[-1]) < float(bb_upper.dropna().iloc[-1])
            if touched_upper and curr_below:
                triggers_hit.append("BB_upper_rejection")

        return len(triggers_hit), triggers_hit

    def check_confirmations(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        open_: Optional[pd.Series] = None,
        vix_rising_intraday: bool = False,
        put_call_above_1: bool = False,
        ad_falling: bool = False,
    ) -> tuple:
        """Level 3: Check confirmation conditions (min 2 of 4)."""
        confirms_hit = []

        # Confirmation 1: Red volume > green (last 3 bars)
        if open_ is not None and len(open_) >= 3:
            recent_close = close.dropna().tail(3)
            recent_open = open_.dropna().tail(3)
            red_bars = sum(1 for c, o in zip(recent_close, recent_open) if c < o)
            if red_bars >= 2:
                confirms_hit.append("red_volume_dominant")

        # Confirmation 2: VIX rising > 5% intraday
        if vix_rising_intraday:
            confirms_hit.append("vix_rising_5pct_intraday")

        # Confirmation 3: Put/Call > 1.0
        if put_call_above_1:
            confirms_hit.append("put_call_above_1")

        # Confirmation 4: A/D line falling
        if ad_falling:
            confirms_hit.append("ad_line_falling")

        return len(confirms_hit), confirms_hit

    def generate(
        self,
        symbol: str,
        daily_data: pd.DataFrame,
        hourly_data: Optional[pd.DataFrame] = None,
        regime_score: int = 3,
        regime: str = "BEAR",
        vix: float = 25.0,
        vix_rising: bool = True,
        spy_below_50sma: bool = True,
        panic_rebound: bool = False,
        vix_rising_intraday: bool = False,
        put_call_above_1: bool = False,
        ad_falling: bool = False,
    ) -> Optional[Signal]:
        """Generate SHORT signal if all 3 levels pass.

        Short executed via SPXS or SH — NOT naked short.
        Size is 1/3 of normal long position.
        """
        data = hourly_data if hourly_data is not None and len(hourly_data) > 50 else daily_data
        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]
        open_ = data.get("Open")

        # Level 1: Setup
        setup_valid, setup_reasons = self.check_setup(
            regime_score=regime_score,
            vix=vix,
            vix_rising=vix_rising,
            spy_below_50sma=spy_below_50sma,
            panic_rebound=panic_rebound,
        )

        if not setup_valid:
            logger.debug("SHORT setup failed: %s", "; ".join(setup_reasons))
            return None

        # Level 2: Triggers
        trigger_count, triggers_hit = self.check_triggers(
            close, high, low, volume, open_
        )

        if trigger_count < self.min_triggers:
            return None

        # Level 3: Confirmations
        confirm_count, confirms_hit = self.check_confirmations(
            close, high, low, volume, open_,
            vix_rising_intraday=vix_rising_intraday,
            put_call_above_1=put_call_above_1,
            ad_falling=ad_falling,
        )

        if confirm_count < self.min_confirms:
            return None

        entry_price = float(close.dropna().iloc[-1])
        stop_price = entry_price * (1 + self.stop_pct)   # stop above entry for short
        target1 = entry_price * (1 - self.target1_pct)
        target2 = entry_price * (1 - self.target2_pct)

        rr_ratio = self.target1_pct / self.stop_pct
        if rr_ratio < self.min_rr:
            return None

        signal = Signal(
            direction=SignalDirection.SHORT,
            symbol=symbol,
            entry_price=entry_price,
            stop_price=stop_price,
            target1=target1,
            target2=target2,
            target3=target2 * 0.97,  # T3 for short
            timestamp=datetime.now(),
            regime_score=regime_score,
            regime=regime,
            setup_valid=True,
            trigger_count=trigger_count,
            confirm_count=confirm_count,
            rr_ratio=rr_ratio,
            triggers_hit=triggers_hit,
            confirms_hit=confirms_hit,
        )

        logger.info(
            "SHORT signal: %s @ %.2f | Stop: %.2f | T1: %.2f | "
            "Triggers: %d | Confirms: %d",
            symbol, entry_price, stop_price, target1,
            trigger_count, confirm_count
        )

        return signal

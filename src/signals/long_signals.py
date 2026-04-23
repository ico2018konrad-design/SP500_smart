"""Long signal generation — 3-level check (Setup, Trigger, Confirmation)."""
import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd

from src.signals.signal_types import Signal, SignalDirection, SignalStrength
from src.signals.indicators import (
    calc_rsi, calc_macd, calc_stochastic, calc_bollinger_bands,
    calc_atr, calc_atr_pct, calc_ema, calc_vwap,
    is_rsi_crossing_up, is_price_touching_ema, is_macd_hist_rising,
    is_price_bb_lower, is_volume_elevated, is_price_above_vwap,
)
from src.regime.regime_types import Regime

logger = logging.getLogger(__name__)


class LongSignalGenerator:
    """Generates LONG entry signals using 3-level check.

    Level 1 — SETUP (ALL required):
        - Regime score >= 8
        - SPY > 200 SMA
        - No major event in 24h
        - ATR(14) < 3%
        - No opposing SHORT position open

    Level 2 — TRIGGER (min 3 of 5):
        1. RSI(14) < 40, crossing up
        2. Price touches 50 EMA and bounces
        3. MACD histogram rising from negative
        4. Stochastic < 30, bullish cross
        5. BB lower touch + green candle

    Level 3 — CONFIRMATION (min 2 of 4):
        1. Volume > 1.3x 20-bar avg
        2. Price above VWAP
        3. A/D line rising (proxy: RSP/SPY)
        4. TICK index > +500 (proxy: breadth)
    """

    def __init__(
        self,
        min_regime_score: int = 8,
        max_atr_pct: float = 0.03,
        min_triggers: int = 3,
        min_confirms: int = 2,
        stop_pct: float = 0.015,
        target1_pct: float = 0.020,
        target2_pct: float = 0.045,
        target3_pct: float = 0.080,
        min_rr: float = 2.0,
    ):
        self.min_regime_score = min_regime_score
        self.max_atr_pct = max_atr_pct
        self.min_triggers = min_triggers
        self.min_confirms = min_confirms
        self.stop_pct = stop_pct
        self.target1_pct = target1_pct
        self.target2_pct = target2_pct
        self.target3_pct = target3_pct
        self.min_rr = min_rr

    def check_setup(
        self,
        regime_score: int,
        spy_above_200sma: bool,
        atr_pct: float,
        has_major_event: bool = False,
        has_short_position: bool = False,
    ) -> tuple:
        """Level 1: Check all setup conditions.

        Returns (is_valid, reasons)
        """
        reasons = []

        if regime_score < self.min_regime_score:
            reasons.append(f"regime_score {regime_score} < {self.min_regime_score}")

        if not spy_above_200sma:
            reasons.append("SPY below 200 SMA")

        if has_major_event:
            reasons.append("major event within 24h blackout")

        if atr_pct >= self.max_atr_pct:
            reasons.append(f"ATR {atr_pct:.1%} >= {self.max_atr_pct:.1%} (too chaotic)")

        if has_short_position:
            reasons.append("opposing SHORT position open")

        is_valid = len(reasons) == 0
        return is_valid, reasons

    def check_triggers(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        open_: Optional[pd.Series] = None,
    ) -> tuple:
        """Level 2: Check trigger conditions (min 3 of 5).

        Returns (trigger_count, triggers_hit)
        """
        triggers_hit = []

        # Compute indicators
        rsi = calc_rsi(close, 14)
        ema50 = calc_ema(close, 50)
        _, _, macd_hist = calc_macd(close)
        stoch_k, stoch_d = calc_stochastic(high, low, close)
        bb_upper, bb_mid, bb_lower = calc_bollinger_bands(close)

        # Trigger 1: RSI < 40, crossing up
        if len(rsi) > 3:
            current_rsi = float(rsi.iloc[-1])
            if current_rsi < 40 and is_rsi_crossing_up(rsi, 40):
                triggers_hit.append("RSI_crossup_40")
            elif current_rsi < 40:
                # Even if not a fresh cross, oversold RSI counts
                triggers_hit.append("RSI_oversold_40")

        # Trigger 2: Price touches 50 EMA and bounces
        if len(ema50) > 3 and len(close) > 3:
            if is_price_touching_ema(close, ema50):
                # Check bounce: price moving up from EMA touch
                if float(close.iloc[-1]) >= float(close.iloc[-2]):
                    triggers_hit.append("EMA50_bounce")

        # Trigger 3: MACD histogram rising from negative
        if len(macd_hist) > 3:
            recent_hist = macd_hist.dropna().tail(3)
            if len(recent_hist) >= 2:
                is_neg_rising = (
                    float(recent_hist.iloc[-2]) < 0 and
                    float(recent_hist.iloc[-1]) > float(recent_hist.iloc[-2])
                )
                if is_neg_rising:
                    triggers_hit.append("MACD_hist_rising_neg")

        # Trigger 4: Stochastic < 30, bullish cross (%K crosses above %D)
        if len(stoch_k) > 3 and len(stoch_d) > 3:
            curr_k = float(stoch_k.dropna().iloc[-1])
            prev_k = float(stoch_k.dropna().iloc[-2])
            curr_d = float(stoch_d.dropna().iloc[-1])
            prev_d = float(stoch_d.dropna().iloc[-2])
            if curr_k < 30 and prev_k <= prev_d and curr_k > curr_d:
                triggers_hit.append("Stoch_bullish_cross_30")
            elif curr_k < 30:
                triggers_hit.append("Stoch_oversold_30")

        # Trigger 5: BB lower touch + green candle
        if len(bb_lower) > 1 and len(close) > 1:
            if is_price_bb_lower(close, bb_lower):
                # Green candle: close > open
                if open_ is not None and len(open_) > 0:
                    is_green = float(close.iloc[-1]) > float(open_.iloc[-1])
                else:
                    is_green = float(close.iloc[-1]) > float(close.iloc[-2])
                if is_green:
                    triggers_hit.append("BB_lower_green_candle")

        return len(triggers_hit), triggers_hit

    def check_confirmations(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        ad_rising: bool = True,
        tick_above_500: bool = True,
    ) -> tuple:
        """Level 3: Check confirmation conditions (min 2 of 4).

        Returns (confirm_count, confirms_hit)
        """
        confirms_hit = []

        # Confirmation 1: Volume > 1.3x average
        if is_volume_elevated(volume, multiplier=1.3, lookback=20):
            confirms_hit.append("volume_elevated")

        # Confirmation 2: Price above VWAP
        vwap = calc_vwap(high, low, close, volume)
        if is_price_above_vwap(close, vwap):
            confirms_hit.append("price_above_vwap")

        # Confirmation 3: A/D line rising (proxy)
        if ad_rising:
            confirms_hit.append("ad_line_rising")

        # Confirmation 4: TICK > +500 (proxy: passed as parameter)
        if tick_above_500:
            confirms_hit.append("tick_above_500")

        return len(confirms_hit), confirms_hit

    def generate(
        self,
        symbol: str,
        daily_data: pd.DataFrame,
        hourly_data: Optional[pd.DataFrame] = None,
        regime_score: int = 8,
        regime: str = "BULL",
        spy_above_200sma: bool = True,
        has_major_event: bool = False,
        has_short_position: bool = False,
        breadth_rising: bool = True,
        tick_above_500: bool = True,
    ) -> Optional[Signal]:
        """Generate LONG signal if all 3 levels pass.

        Args:
            symbol: Trading symbol (SPY, UPRO)
            daily_data: Daily OHLCV DataFrame
            hourly_data: Hourly OHLCV DataFrame for trigger check
            regime_score: Current regime score (0-11)
            regime: Regime string
            spy_above_200sma: Whether SPY is above 200 SMA
            has_major_event: Whether major economic event within 24h
            has_short_position: Whether opposing short is open
            breadth_rising: Whether A/D line / breadth is rising
            tick_above_500: Whether NYSE TICK > 500

        Returns:
            Signal if all levels pass, None otherwise
        """
        # Use hourly data for triggers if available, else daily
        data = hourly_data if hourly_data is not None and len(hourly_data) > 50 else daily_data

        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]
        open_ = data.get("Open")

        # Calculate ATR%
        atr_pct_series = calc_atr_pct(high, low, close)
        current_atr_pct = float(atr_pct_series.dropna().iloc[-1]) if len(atr_pct_series.dropna()) > 0 else 0.02

        # Level 1: Setup
        setup_valid, setup_reasons = self.check_setup(
            regime_score=regime_score,
            spy_above_200sma=spy_above_200sma,
            atr_pct=current_atr_pct,
            has_major_event=has_major_event,
            has_short_position=has_short_position,
        )

        if not setup_valid:
            logger.debug("LONG setup failed: %s", "; ".join(setup_reasons))
            return None

        # Level 2: Triggers
        trigger_count, triggers_hit = self.check_triggers(close, high, low, volume, open_)

        if trigger_count < self.min_triggers:
            logger.debug(
                "LONG trigger check: %d/%d (%s)",
                trigger_count, self.min_triggers, triggers_hit
            )
            return None

        # Level 3: Confirmations
        confirm_count, confirms_hit = self.check_confirmations(
            close, high, low, volume,
            ad_rising=breadth_rising,
            tick_above_500=tick_above_500,
        )

        if confirm_count < self.min_confirms:
            logger.debug(
                "LONG confirmation check: %d/%d (%s)",
                confirm_count, self.min_confirms, confirms_hit
            )
            return None

        # All levels passed — create signal
        entry_price = float(close.dropna().iloc[-1])
        stop_price = entry_price * (1 - self.stop_pct)
        target1 = entry_price * (1 + self.target1_pct)
        target2 = entry_price * (1 + self.target2_pct)
        target3 = entry_price * (1 + self.target3_pct)

        rr_ratio = self.target1_pct / self.stop_pct

        if rr_ratio < self.min_rr:
            logger.debug("LONG R:R %.2f below minimum %.2f", rr_ratio, self.min_rr)
            return None

        # Determine signal strength
        total_quality = trigger_count + confirm_count
        if total_quality >= 7:
            strength = SignalStrength.STRONG
        elif total_quality >= 5:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        signal = Signal(
            direction=SignalDirection.LONG,
            symbol=symbol,
            entry_price=entry_price,
            stop_price=stop_price,
            target1=target1,
            target2=target2,
            target3=target3,
            timestamp=datetime.now(),
            regime_score=regime_score,
            regime=regime,
            setup_valid=True,
            trigger_count=trigger_count,
            confirm_count=confirm_count,
            rr_ratio=rr_ratio,
            triggers_hit=triggers_hit,
            confirms_hit=confirms_hit,
            strength=strength,
        )

        logger.info(
            "LONG signal generated: %s @ %.2f | Stop: %.2f | T1: %.2f | "
            "Triggers: %d/5 | Confirms: %d/4 | Regime: %s",
            symbol, entry_price, stop_price, target1,
            trigger_count, confirm_count, regime
        )

        return signal

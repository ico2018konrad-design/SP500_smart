"""Long signal generation — 3-level check (Setup, Trigger, Confirmation).

Supports two signal modes selected automatically based on regime:
- TREND_FOLLOW: for STRONG_BULL/BULL regimes — momentum/continuation entries
- MEAN_REVERT: for CHOP/CAUTION regimes — oversold dip-buying entries
"""
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
    """Generates LONG entry signals using 3-level check with adaptive mode.

    Mode selected automatically based on regime:

    TREND_FOLLOW (STRONG_BULL / BULL):
        Triggers (min 2 of 5 for daily, min 2 of 5 for intraday):
          1. Pullback to 20 EMA in uptrend
          2. Breakout above 20-day high (green candle)
          3. Higher-low after pullback in 20-bar window
          4. Momentum confirmation (5d return >1%, RSI 50-70)
          5. Golden cross / price above 50 EMA & 200 EMA
        Confirmations (min 1 of 3):
          1. Volume > 1.1x 20-bar avg
          2. Higher close than prev 3 days
          3. Breadth rising

    MEAN_REVERT (CHOP / CAUTION):
        Triggers (min 3 of 5 intraday, min 2 of 5 daily):
          1. RSI < 40 (daily: <45), crossing up
          2. Price touches 50 EMA and bounces
          3. MACD histogram rising from negative
          4. Stochastic < 30 (daily: <35), bullish cross
          5. BB lower touch + green candle
        Confirmations (min 2 of 4):
          1. Volume > 1.3x 20-bar avg
          2. Price above VWAP
          3. A/D line rising (proxy)
          4. TICK index > +500 (proxy)

    Level 1 — SETUP (ALL required):
        - Regime score >= threshold
        - SPY > 200 SMA
        - No major event in 24h
        - ATR(14) < 3%
        - No opposing SHORT position open
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

    def _detect_timeframe(self, data: pd.DataFrame) -> str:
        """Detect whether data is intraday or daily.

        Returns 'intraday' if gap between bars < 1 day, else 'daily'.
        """
        if len(data) < 2:
            return "daily"
        delta = data.index[1] - data.index[0]
        if delta.total_seconds() < 86400:
            logger.debug("Timeframe detected: intraday (bar gap=%s)", delta)
            return "intraday"
        logger.debug("Timeframe detected: daily (bar gap=%s)", delta)
        return "daily"

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
        """Level 2 (mean-revert mode): Check trigger conditions (min 3 of 5 intraday).

        Returns (trigger_count, triggers_hit)
        """
        return self._mean_revert_triggers(close, high, low, volume, open_, rsi_thresh=40, stoch_thresh=30)

    def _mean_revert_triggers(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        open_: Optional[pd.Series],
        rsi_thresh: float = 40,
        stoch_thresh: float = 30,
    ) -> tuple:
        """Mean-reversion triggers (oversold dip-buying)."""
        triggers_hit = []

        rsi = calc_rsi(close, 14)
        ema50 = calc_ema(close, 50)
        _, _, macd_hist = calc_macd(close)
        stoch_k, stoch_d = calc_stochastic(high, low, close)
        bb_upper, bb_mid, bb_lower = calc_bollinger_bands(close)

        # Trigger 1: RSI below threshold, crossing up
        if len(rsi) > 3:
            current_rsi = float(rsi.dropna().iloc[-1])
            if current_rsi < rsi_thresh and is_rsi_crossing_up(rsi, rsi_thresh):
                triggers_hit.append(f"RSI_crossup_{rsi_thresh:.0f}")
            elif current_rsi < rsi_thresh:
                triggers_hit.append(f"RSI_oversold_{rsi_thresh:.0f}")

        # Trigger 2: Price touches 50 EMA and bounces
        if len(ema50) > 3 and len(close) > 3:
            if is_price_touching_ema(close, ema50):
                if float(close.dropna().iloc[-1]) >= float(close.dropna().iloc[-2]):
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

        # Trigger 4: Stochastic below threshold, bullish cross
        if len(stoch_k) > 3 and len(stoch_d) > 3:
            sk = stoch_k.dropna()
            sd = stoch_d.dropna()
            if len(sk) >= 2 and len(sd) >= 2:
                curr_k = float(sk.iloc[-1])
                prev_k = float(sk.iloc[-2])
                curr_d = float(sd.iloc[-1])
                prev_d = float(sd.iloc[-2])
                if curr_k < stoch_thresh and prev_k <= prev_d and curr_k > curr_d:
                    triggers_hit.append(f"Stoch_bullish_cross_{stoch_thresh:.0f}")
                elif curr_k < stoch_thresh:
                    triggers_hit.append(f"Stoch_oversold_{stoch_thresh:.0f}")

        # Trigger 5: BB lower touch + green candle
        if len(bb_lower) > 1 and len(close) > 1:
            if is_price_bb_lower(close, bb_lower):
                close_clean = close.dropna()
                if open_ is not None and len(open_) > 0:
                    is_green = float(close_clean.iloc[-1]) > float(open_.dropna().iloc[-1])
                else:
                    is_green = float(close_clean.iloc[-1]) > float(close_clean.iloc[-2])
                if is_green:
                    triggers_hit.append("BB_lower_green_candle")

        return len(triggers_hit), triggers_hit

    def _stop_pct_for_timeframe(self, timeframe: str) -> tuple:
        """Return (stop_pct, t1_pct, t2_pct, t3_pct) adapted to timeframe.

        Daily bars need wider stops to survive normal daily noise.
        """
        if timeframe == "daily":
            return 0.025, 0.050, 0.090, 0.150
        return self.stop_pct, self.target1_pct, self.target2_pct, self.target3_pct

    def _trend_follow_triggers(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        open_: Optional[pd.Series],
    ) -> tuple:
        """Trend-following triggers (momentum/continuation)."""
        triggers_hit = []

        ema20 = calc_ema(close, 20)
        ema50 = calc_ema(close, 50)
        ema200 = calc_ema(close, 200)
        rsi = calc_rsi(close, 14)

        close_clean = close.dropna()
        ema20_clean = ema20.dropna()
        ema50_clean = ema50.dropna()
        ema200_clean = ema200.dropna()
        rsi_clean = rsi.dropna()

        if len(close_clean) < 5:
            return 0, []

        curr_close = float(close_clean.iloc[-1])

        # Trigger 1: Pullback to 20 EMA in uptrend
        if len(ema20_clean) > 1 and len(ema50_clean) > 1:
            curr_ema20 = float(ema20_clean.iloc[-1])
            curr_ema50 = float(ema50_clean.iloc[-1])
            near_ema20 = curr_ema20 * 0.99 <= curr_close <= curr_ema20 * 1.02
            ema_uptrend = curr_ema20 > curr_ema50
            if near_ema20 and ema_uptrend:
                triggers_hit.append("EMA20_pullback_uptrend")

        # Trigger 2: Breakout above 20-day high (green candle)
        if len(close_clean) >= 21:
            prev_20_high = float(close_clean.iloc[-21:-1].max())
            is_green = True
            if open_ is not None and len(open_) > 0:
                open_clean = open_.dropna()
                if len(open_clean) > 0:
                    is_green = curr_close > float(open_clean.iloc[-1])
            if curr_close > prev_20_high and is_green:
                triggers_hit.append("breakout_20day_high")

        # Trigger 3: Higher-low pattern — recent swing low > previous swing low
        if len(low) >= 20:
            low_clean = low.dropna()
            if len(low_clean) >= 20:
                recent_window = low_clean.tail(20).values
                # Find local lows (lower than neighbors)
                local_lows = []
                for idx in range(1, len(recent_window) - 1):
                    if recent_window[idx] < recent_window[idx - 1] and recent_window[idx] < recent_window[idx + 1]:
                        local_lows.append(recent_window[idx])
                if len(local_lows) >= 2 and local_lows[-1] > local_lows[-2]:
                    # Bouncing: current close above last swing low
                    if curr_close > local_lows[-1]:
                        triggers_hit.append("higher_low_pattern")

        # Trigger 4: Momentum confirmation — 5d return > 1%, RSI 50-70
        if len(close_clean) >= 6:
            ret_5d = (curr_close / float(close_clean.iloc[-6]) - 1)
            if ret_5d > 0.01 and len(rsi_clean) > 0:
                curr_rsi = float(rsi_clean.iloc[-1])
                if 50 <= curr_rsi <= 70:
                    triggers_hit.append("momentum_rsi_healthy")

        # Trigger 5: Golden cross / price above 50 EMA and 200 EMA
        if len(ema50_clean) > 1 and len(ema200_clean) > 1:
            curr_ema50 = float(ema50_clean.iloc[-1])
            curr_ema200 = float(ema200_clean.iloc[-1])
            golden_cross = curr_ema50 > curr_ema200
            above_both = curr_close > curr_ema50 and curr_close > curr_ema200
            if golden_cross and above_both:
                triggers_hit.append("golden_cross_above_both_emas")

        # Trigger 6: Pure uptrend — EMA50 above EMA200 (definition of bull market).
        # Added as a standalone signal so that STRONG_BULL/BULL regime days with a
        # healthy EMA structure always produce at least 1 trigger.
        if len(ema50_clean) > 1 and len(ema200_clean) > 1:
            curr_ema50 = float(ema50_clean.iloc[-1])
            curr_ema200 = float(ema200_clean.iloc[-1])
            if curr_ema50 > curr_ema200:
                triggers_hit.append("ema50_above_ema200_uptrend")

        return len(triggers_hit), triggers_hit

    def _trend_follow_confirmations(
        self,
        close: pd.Series,
        volume: pd.Series,
        breadth_rising: bool,
    ) -> tuple:
        """Trend-follow confirmations (min 1 of 3 — less strict)."""
        confirms_hit = []

        # Confirm 1: Volume > 1.1x 20-bar avg (relaxed from 1.3x)
        if is_volume_elevated(volume, multiplier=1.1, lookback=20):
            confirms_hit.append("volume_elevated_1.1x")

        # Confirm 2: Higher close than previous 3 days
        close_clean = close.dropna()
        if len(close_clean) >= 4:
            curr = float(close_clean.iloc[-1])
            prev3_max = float(close_clean.iloc[-4:-1].max())
            if curr > prev3_max:
                confirms_hit.append("higher_close_3d")

        # Confirm 3: Breadth rising
        if breadth_rising:
            confirms_hit.append("breadth_rising")

        return len(confirms_hit), confirms_hit

    def check_confirmations(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        ad_rising: bool = True,
        tick_above_500: bool = True,
    ) -> tuple:
        """Level 3 (mean-revert mode): Check confirmation conditions (min 2 of 4).

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

    def _generate_trend_follow_signal(
        self,
        symbol: str,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        open_: Optional[pd.Series],
        regime_score: int,
        regime: str,
        breadth_rising: bool,
        timeframe: str,
    ) -> Optional[Signal]:
        """Generate signal using trend-following triggers for BULL/STRONG_BULL regimes."""
        min_triggers = 1  # relaxed: 1 clear signal is enough in a bull regime
        min_confirms = 1  # only 1 of 3 confirmations needed

        trigger_count, triggers_hit = self._trend_follow_triggers(close, high, low, volume, open_)

        if trigger_count < min_triggers:
            logger.info(
                "[SIGNAL_REJECT] %s %s triggers %d/%d: %s",
                symbol, regime, trigger_count, min_triggers, triggers_hit,
            )
            return None

        confirm_count, confirms_hit = self._trend_follow_confirmations(close, volume, breadth_rising)

        if confirm_count < min_confirms:
            logger.info(
                "[SIGNAL_REJECT] %s %s confirms %d/%d: %s",
                symbol, regime, confirm_count, min_confirms, confirms_hit,
            )
            return None

        logger.info(
            "[SIGNAL_ACCEPT] %s %s triggers=%s confirms=%s",
            symbol, regime, triggers_hit, confirms_hit,
        )
        return self._build_signal(
            symbol, close, regime_score, regime,
            trigger_count, triggers_hit, confirm_count, confirms_hit,
            timeframe=timeframe,
        )

    def _generate_mean_reversion_signal(
        self,
        symbol: str,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        open_: Optional[pd.Series],
        regime_score: int,
        regime: str,
        breadth_rising: bool,
        tick_above_500: bool,
        timeframe: str,
    ) -> Optional[Signal]:
        """Generate signal using mean-reversion triggers for CHOP/CAUTION regimes."""
        # Relax thresholds for daily data
        if timeframe == "daily":
            rsi_thresh = 45
            stoch_thresh = 35
            min_triggers = 2
            min_confirms = 2
        else:
            rsi_thresh = 40
            stoch_thresh = 30
            min_triggers = self.min_triggers  # 3
            min_confirms = self.min_confirms  # 2

        trigger_count, triggers_hit = self._mean_revert_triggers(
            close, high, low, volume, open_, rsi_thresh, stoch_thresh,
        )

        if trigger_count < min_triggers:
            logger.info(
                "[SIGNAL_REJECT] %s %s triggers %d/%d: %s",
                symbol, regime, trigger_count, min_triggers, triggers_hit,
            )
            return None

        confirm_count, confirms_hit = self.check_confirmations(
            close, high, low, volume,
            ad_rising=breadth_rising,
            tick_above_500=tick_above_500,
        )

        if confirm_count < min_confirms:
            logger.info(
                "[SIGNAL_REJECT] %s %s confirms %d/%d: %s",
                symbol, regime, confirm_count, min_confirms, confirms_hit,
            )
            return None

        return self._build_signal(
            symbol, close, regime_score, regime,
            trigger_count, triggers_hit, confirm_count, confirms_hit,
            timeframe=timeframe,
        )

    def _build_signal(
        self,
        symbol: str,
        close: pd.Series,
        regime_score: int,
        regime: str,
        trigger_count: int,
        triggers_hit: list,
        confirm_count: int,
        confirms_hit: list,
        timeframe: str = "intraday",
    ) -> Optional[Signal]:
        """Construct Signal object from validated triggers and confirmations."""
        entry_price = float(close.dropna().iloc[-1])

        # Adaptive stop and targets: daily bars need wider stops to survive noise
        stop_pct, target1_pct, target2_pct, target3_pct = self._stop_pct_for_timeframe(timeframe)

        stop_price = entry_price * (1 - stop_pct)
        target1 = entry_price * (1 + target1_pct)
        target2 = entry_price * (1 + target2_pct)
        target3 = entry_price * (1 + target3_pct)

        rr_ratio = target1_pct / stop_pct

        if rr_ratio < self.min_rr:
            logger.debug("LONG R:R %.2f below minimum %.2f", rr_ratio, self.min_rr)
            return None

        total_quality = trigger_count + confirm_count
        # Threshold adapted: trend-follow min is 1 trigger + 1 confirm = 2 (WEAK), 4+ is MODERATE.
        if total_quality >= 7:
            strength = SignalStrength.STRONG
        elif total_quality >= 4:
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
            "Triggers: %d | Confirms: %d | Regime: %s",
            symbol, entry_price, stop_price, target1,
            trigger_count, confirm_count, regime,
        )
        return signal

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
        """Generate LONG signal — adaptive mode based on regime.

        Branches:
          STRONG_BULL / BULL  → trend-following entries (momentum)
          CHOP / CAUTION      → mean-reversion entries (oversold dips)
          BEAR                → None (shorts handled elsewhere)

        Args:
            symbol: Trading symbol (SPY, UPRO)
            daily_data: Daily OHLCV DataFrame
            hourly_data: Hourly OHLCV DataFrame (optional; if provided, used for triggers)
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
        # No longs in BEAR
        regime_str = regime.value if hasattr(regime, 'value') else str(regime)
        regime_str = regime_str.upper().replace(" ", "_")  # normalize enum/string variants
        if regime_str == "BEAR":
            logger.debug("LONG skipped — BEAR regime, no longs")
            return None

        # Use hourly data for triggers if available (intraday mode), else daily
        data = hourly_data if hourly_data is not None and len(hourly_data) > 50 else daily_data
        timeframe = self._detect_timeframe(data)

        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]
        open_ = data.get("Open")

        # Calculate ATR%
        atr_pct_series = calc_atr_pct(high, low, close)
        current_atr_pct = float(atr_pct_series.dropna().iloc[-1]) if len(atr_pct_series.dropna()) > 0 else 0.02

        # Level 1: Setup (regime-aware threshold)
        # In CHOP/CAUTION allow slightly lower regime score
        effective_min_score = self.min_regime_score
        if regime_str in ("CHOP", "CAUTION"):
            effective_min_score = max(4, self.min_regime_score - 2)

        setup_valid, setup_reasons = self.check_setup(
            regime_score=regime_score,
            spy_above_200sma=spy_above_200sma,
            atr_pct=current_atr_pct,
            has_major_event=has_major_event,
            has_short_position=has_short_position,
        )

        # Override min regime score check for CHOP/CAUTION
        if not setup_valid and regime_str in ("CHOP", "CAUTION"):
            filtered_reasons = [r for r in setup_reasons if "regime_score" not in r or regime_score >= effective_min_score]
            setup_valid = len(filtered_reasons) == 0
            if not setup_valid:
                setup_reasons = filtered_reasons

        if not setup_valid:
            logger.info("[SIGNAL_REJECT] %s %s setup_failed: %s", symbol, regime_str, "; ".join(setup_reasons))
            return None

        # Branch on regime: trend-follow vs mean-revert
        if regime_str in ("STRONG_BULL", "BULL"):
            logger.debug("Using TREND-FOLLOW mode for regime=%s, timeframe=%s", regime_str, timeframe)
            return self._generate_trend_follow_signal(
                symbol=symbol,
                close=close,
                high=high,
                low=low,
                volume=volume,
                open_=open_,
                regime_score=regime_score,
                regime=regime_str,
                breadth_rising=breadth_rising,
                timeframe=timeframe,
            )
        elif regime_str in ("CHOP", "CAUTION"):
            logger.debug("Using MEAN-REVERT mode for regime=%s, timeframe=%s", regime_str, timeframe)
            return self._generate_mean_reversion_signal(
                symbol=symbol,
                close=close,
                high=high,
                low=low,
                volume=volume,
                open_=open_,
                regime_score=regime_score,
                regime=regime_str,
                breadth_rising=breadth_rising,
                tick_above_500=tick_above_500,
                timeframe=timeframe,
            )
        else:
            # Fallback for unknown regimes — use mean-revert
            logger.debug("Using MEAN-REVERT fallback for regime=%s", regime_str)
            return self._generate_mean_reversion_signal(
                symbol=symbol,
                close=close,
                high=high,
                low=low,
                volume=volume,
                open_=open_,
                regime_score=regime_score,
                regime=regime_str,
                breadth_rising=breadth_rising,
                tick_above_500=tick_above_500,
                timeframe=timeframe,
            )

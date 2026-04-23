"""Regime Detector — 11-indicator market regime scoring system.

Each indicator contributes 0 or 1 point to total score (0-11).
Score maps to regime: STRONG_BULL/BULL/CHOP/CAUTION/BEAR.
"""
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.regime.regime_types import Regime, RegimeResult, REGIME_LEVERAGE, REGIME_RISK_PCT, score_to_regime

logger = logging.getLogger(__name__)


class RegimeDetector:
    """11-indicator regime detector.

    Indicators:
    TREND (4 pts):
        1. SPY > 200 SMA
        2. SPY > 50 SMA
        3. 200 SMA slope positive (30d)
        4. Higher highs + higher lows structure

    MOMENTUM (2 pts):
        5. Weekly MACD bullish
        6. ADX(14) > 25 with DI+ > DI-

    VOLATILITY (2 pts):
        7. VIX < 20
        8. VIX term structure in contango

    BREADTH (1 pt):
        9. % stocks above 50 SMA > 55%

    CREDIT/MACRO (2 pts):
        10. HY credit spreads < 400 bps
        11. Yield curve not freshly inverted
    """

    def __init__(
        self,
        sma200_period: int = 200,
        sma50_period: int = 50,
        adx_period: int = 14,
        vix_threshold: float = 20.0,
        breadth_threshold: float = 0.55,
        hy_spread_threshold: float = 400.0,
        slope_lookback: int = 30,
        hhhl_lookback: int = 20,
    ):
        self.sma200_period = sma200_period
        self.sma50_period = sma50_period
        self.adx_period = adx_period
        self.vix_threshold = vix_threshold
        self.breadth_threshold = breadth_threshold
        self.hy_spread_threshold = hy_spread_threshold
        self.slope_lookback = slope_lookback
        self.hhhl_lookback = hhhl_lookback

    def detect(
        self,
        spy_daily: pd.DataFrame,
        vix_daily: pd.DataFrame,
        breadth_pct: float = 0.55,
        hy_spread_bps: float = 350.0,
        yield_curve_freshly_inverted: bool = False,
        vix_9d: Optional[float] = None,
        vx1: Optional[float] = None,
    ) -> RegimeResult:
        """Run full 11-indicator regime detection.

        Args:
            spy_daily: SPY OHLCV DataFrame (at least 200 trading days)
            vix_daily: VIX daily DataFrame
            breadth_pct: % of S&P 500 stocks above 50 SMA (0.0-1.0)
            hy_spread_bps: HY credit spread in basis points
            yield_curve_freshly_inverted: True if 10Y-2Y recently crossed below 0
            vix_9d: 9-day VIX (for term structure)
            vx1: 1-month VIX futures (for term structure)

        Returns:
            RegimeResult with score, regime, and indicator details
        """
        scores = {}
        details = {}

        # ── TREND INDICATORS ─────────────────────────────────────────────────
        close = spy_daily["Close"] if "Close" in spy_daily else spy_daily.iloc[:, 0]
        close = close.dropna()

        # 1. SPY > 200 SMA
        spy_above_200sma = False
        if len(close) >= self.sma200_period:
            sma200 = close.rolling(self.sma200_period).mean()
            spy_above_200sma = bool(close.iloc[-1] > sma200.iloc[-1])
            details["sma200"] = float(sma200.iloc[-1])
            details["spy_price"] = float(close.iloc[-1])
        scores["spy_above_200sma"] = int(spy_above_200sma)

        # 2. SPY > 50 SMA
        spy_above_50sma = False
        if len(close) >= self.sma50_period:
            sma50 = close.rolling(self.sma50_period).mean()
            spy_above_50sma = bool(close.iloc[-1] > sma50.iloc[-1])
            details["sma50"] = float(sma50.iloc[-1])
        scores["spy_above_50sma"] = int(spy_above_50sma)

        # 3. 200 SMA slope positive (30d)
        sma200_slope_positive = False
        if len(close) >= self.sma200_period + self.slope_lookback:
            sma200_now = close.rolling(self.sma200_period).mean().iloc[-1]
            sma200_prev = close.rolling(self.sma200_period).mean().iloc[-self.slope_lookback - 1]
            if not pd.isna(sma200_prev) and not pd.isna(sma200_now):
                sma200_slope_positive = bool(sma200_now > sma200_prev)
                details["sma200_slope"] = float(sma200_now - sma200_prev)
        scores["sma200_slope_positive"] = int(sma200_slope_positive)

        # 4. Higher highs + higher lows (last N bars)
        higher_highs_lows = self._check_higher_highs_lows(spy_daily)
        scores["higher_highs_lows"] = int(higher_highs_lows)

        # ── MOMENTUM INDICATORS ──────────────────────────────────────────────
        # 5. Weekly MACD bullish
        macd_weekly_bullish = self._check_weekly_macd(spy_daily)
        scores["macd_weekly_bullish"] = int(macd_weekly_bullish)

        # 6. ADX(14) > 25 with DI+ > DI-
        adx_bullish = self._check_adx(spy_daily)
        scores["adx_bullish"] = int(adx_bullish)

        # ── VOLATILITY INDICATORS ────────────────────────────────────────────
        vix_close = vix_daily["Close"] if "Close" in vix_daily.columns else vix_daily.iloc[:, 0]
        vix_close = vix_close.dropna()
        current_vix = float(vix_close.iloc[-1]) if len(vix_close) > 0 else 20.0
        details["vix"] = current_vix

        # 7. VIX < 20
        vix_below_20 = current_vix < self.vix_threshold
        scores["vix_below_20"] = int(vix_below_20)

        # 8. VIX term structure in contango (VIX9D < VIX < VX1)
        vix_term_contango = self._check_vix_contango(current_vix, vix_9d, vx1)
        scores["vix_term_contango"] = int(vix_term_contango)

        # ── BREADTH ──────────────────────────────────────────────────────────
        # 9. % stocks above 50 SMA > 55%
        breadth_ok = breadth_pct > self.breadth_threshold
        scores["breadth_above_55pct"] = int(breadth_ok)
        details["breadth_pct"] = breadth_pct

        # ── CREDIT/MACRO ─────────────────────────────────────────────────────
        # 10. HY credit spreads < 400 bps
        hy_ok = hy_spread_bps < self.hy_spread_threshold
        scores["hy_spread_ok"] = int(hy_ok)
        details["hy_spread_bps"] = hy_spread_bps

        # 11. Yield curve not freshly inverted
        yield_curve_ok = not yield_curve_freshly_inverted
        scores["yield_curve_ok"] = int(yield_curve_ok)
        details["yield_curve_freshly_inverted"] = yield_curve_freshly_inverted

        # ── FINAL SCORE ──────────────────────────────────────────────────────
        total_score = sum(scores.values())
        regime = score_to_regime(total_score)
        max_leverage = REGIME_LEVERAGE[regime]
        risk_pct = REGIME_RISK_PCT[regime]

        logger.info(
            "Regime: %s (score=%d/11) | Leverage: %.1fx | "
            "Trend: %d/4, Mom: %d/2, Vol: %d/2, Breadth: %d/1, Macro: %d/2",
            regime.value, total_score, max_leverage,
            scores.get("spy_above_200sma", 0) + scores.get("spy_above_50sma", 0) +
            scores.get("sma200_slope_positive", 0) + scores.get("higher_highs_lows", 0),
            scores.get("macd_weekly_bullish", 0) + scores.get("adx_bullish", 0),
            scores.get("vix_below_20", 0) + scores.get("vix_term_contango", 0),
            scores.get("breadth_above_55pct", 0),
            scores.get("hy_spread_ok", 0) + scores.get("yield_curve_ok", 0),
        )

        return RegimeResult(
            score=total_score,
            regime=regime,
            max_leverage=max_leverage,
            risk_pct=risk_pct,
            indicator_scores=scores,
            details=details,
            spy_above_200sma=spy_above_200sma,
            spy_above_50sma=spy_above_50sma,
            sma200_slope_positive=sma200_slope_positive,
            higher_highs_lows=higher_highs_lows,
            macd_weekly_bullish=macd_weekly_bullish,
            adx_bullish=adx_bullish,
            vix_below_20=vix_below_20,
            vix_term_contango=vix_term_contango,
            breadth_above_55pct=breadth_ok,
            hy_spread_ok=hy_ok,
            yield_curve_ok=yield_curve_ok,
            timestamp=datetime.now().isoformat(),
        )

    def _check_higher_highs_lows(self, spy_daily: pd.DataFrame) -> bool:
        """Check higher highs and higher lows structure."""
        try:
            high = spy_daily["High"].dropna().tail(self.hhhl_lookback)
            low = spy_daily["Low"].dropna().tail(self.hhhl_lookback)

            if len(high) < 4:
                return False

            mid = len(high) // 2
            # First half vs second half
            recent_high = high.tail(mid).max()
            old_high = high.head(mid).max()
            recent_low = low.tail(mid).min()
            old_low = low.head(mid).min()

            return bool(recent_high > old_high and recent_low > old_low)
        except Exception:
            return False

    def _check_weekly_macd(self, spy_daily: pd.DataFrame) -> bool:
        """Check if weekly MACD is bullish (MACD line > Signal line)."""
        try:
            close = spy_daily["Close"].dropna()
            if len(close) < 60:
                return False

            # Resample to weekly
            weekly = close.resample("W").last().dropna()
            if len(weekly) < 30:
                return False

            ema12 = weekly.ewm(span=12, adjust=False).mean()
            ema26 = weekly.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()

            return bool(macd_line.iloc[-1] > signal_line.iloc[-1])
        except Exception:
            return False

    def _check_adx(self, spy_daily: pd.DataFrame, threshold: float = 25.0) -> bool:
        """Check ADX > threshold with DI+ > DI-."""
        try:
            high = spy_daily["High"].dropna()
            low = spy_daily["Low"].dropna()
            close = spy_daily["Close"].dropna()

            if len(close) < self.adx_period + 1:
                return False

            # Calculate True Range
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)

            # Directional Movement
            up_move = high.diff()
            down_move = -low.diff()

            plus_dm = pd.Series(
                np.where((up_move > down_move) & (up_move > 0), up_move, 0),
                index=high.index
            )
            minus_dm = pd.Series(
                np.where((down_move > up_move) & (down_move > 0), down_move, 0),
                index=high.index
            )

            # Smooth with Wilder's method
            period = self.adx_period
            atr = tr.ewm(alpha=1/period, adjust=False).mean()
            plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
            minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            adx = dx.ewm(alpha=1/period, adjust=False).mean()

            adx_val = float(adx.iloc[-1])
            plus_di_val = float(plus_di.iloc[-1])
            minus_di_val = float(minus_di.iloc[-1])

            return bool(adx_val > threshold and plus_di_val > minus_di_val)
        except Exception:
            return False

    def _check_vix_contango(
        self,
        current_vix: float,
        vix_9d: Optional[float],
        vx1: Optional[float],
    ) -> bool:
        """Check VIX term structure contango: VIX9D < VIX < VX1."""
        if vix_9d is None or vx1 is None:
            # If data not available, assume contango in normal conditions
            return current_vix < 25.0

        return bool(vix_9d < current_vix < vx1)

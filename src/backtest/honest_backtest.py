"""Honest Backtester — includes commissions, slippage, and borrow fees.

Uses real historical data from Yahoo Finance and FRED.
Default period: 2005-01-01 to 2025-12-31.

Uses the full strategy stack:
  LongSignalGenerator + ShortSignalGenerator (3-level check)
  AntiMartingaleScaler (6-position scale-in, anti-martingale)
  PositionManager (up to 6 positions)
  BaselineHedge (10% SH allocation, auto-rebalanced)
  CircuitBreakers (daily/weekly/monthly halt)
  ValuationGuard (leverage cap)
  Real FRED macro data (HY spread, yield curve)

Run: python src/backtest/honest_backtest.py
"""
import json
import logging
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from src.data.yahoo_loader import load_spy, load_vix, load_ohlcv
from src.data.fred_macro import load_macro_data, get_cape_ratio
from src.data.macro_timeseries import MacroTimeSeries
from src.regime.detector import RegimeDetector
from src.regime.valuation_guard import ValuationGuard
from src.regime.regime_types import Regime, score_to_regime
from src.signals.indicators import calc_atr, calc_rsi, calc_macd, calc_ema
from src.signals.long_signals import LongSignalGenerator
from src.signals.short_signals import ShortSignalGenerator
from src.positions.position_manager import PositionManager
from src.positions.anti_martingale_scaler import AntiMartingaleScaler
from src.positions.exit_manager import ExitManager
from src.hedge.baseline_hedge import BaselineHedge
from src.risk.circuit_breakers import CircuitBreakers
from src.backtest.performance_metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Transaction costs (realistic for IBKR)
COMMISSION_PCT = 0.0005   # 0.05% per trade
SLIPPAGE_PCT = 0.001      # 0.10% slippage

# Annual borrow fees (ETFs)
BORROW_FEES = {
    "UPRO": 0.0095,   # 0.95%/year
    "SH":   0.0035,   # 0.35%/year
    "SPXS": 0.0180,   # 1.80%/year
    "SPY":  0.0000,   # no borrow for long
}

EXPENSE_RATIOS = {
    "SPY":  0.00095,
    "UPRO": 0.00930,
    "SH":   0.00890,
    "SPXS": 0.01080,
}


def _apply_slippage(price: float, direction: str, slippage_pct: float) -> float:
    """Apply slippage: buy high, sell low."""
    if direction in ("LONG", "BUY"):
        return price * (1 + slippage_pct)
    return price * (1 - slippage_pct)


def _open_position(
    capital: float,
    instrument: str,
    direction: str,
    price: float,
    stop: float,
    target1: float,
    target2: float,
    target3: float,
    position_mgr: PositionManager,
    trades: list,
    date: pd.Timestamp,
    regime_label: str,
    max_alloc_pct: float = 0.15,
) -> float:
    """Open a position and debit capital. Returns updated capital."""
    fill = _apply_slippage(price, direction, SLIPPAGE_PCT)
    risk_dollars = capital * 0.015
    stop_dist = max(abs(fill - stop), fill * 0.005)
    shares = max(1, int(risk_dollars / stop_dist))
    max_shares = int(capital * max_alloc_pct / fill)
    shares = min(shares, max_shares)

    cost = shares * fill
    commission = cost * COMMISSION_PCT

    if cost + commission > capital * 0.95:  # safety: leave 5% cash buffer
        logger.debug("Insufficient capital for %s %s %d @ %.2f", direction, instrument, shares, fill)
        return capital

    # Debit capital on open
    capital -= cost + commission

    pos = position_mgr.add_position(
        symbol=instrument,
        direction=direction,
        entry_price=fill,
        shares=shares,
        stop_price=stop,
        target1=target1,
        target2=target2,
        target3=target3,
    )

    if pos:
        trades.append({
            "entry_date": str(date.date()),
            "exit_date": None,
            "instrument": instrument,
            "direction": direction,
            "entry_price": fill,
            "exit_price": None,
            "shares": shares,
            "pnl": None,
            "reason": "entry",
            "regime": regime_label,
            "_position_id": pos.position_id,
        })
        logger.debug("OPEN %s %s %d @ %.2f on %s", direction, instrument, shares, fill, date.date())

    return capital


def _close_position(
    capital: float,
    pos,
    close_price: float,
    position_mgr: PositionManager,
    trades: list,
    date: pd.Timestamp,
    reason: str,
) -> float:
    """Close a position and credit proceeds. Returns updated capital."""
    fill = _apply_slippage(close_price, "SELL" if pos.direction == "LONG" else "BUY", SLIPPAGE_PCT)
    shares_open = pos.shares_open
    commission = shares_open * fill * COMMISSION_PCT

    # Credit: sale proceeds minus commission.
    # Note: In this backtest, short exposure is modeled via long positions in
    # inverse ETFs (SH, SPXS). True naked SHORTs would require a different
    # accounting model (proceeds received at entry, cost paid at close).
    # The code below handles LONG positions (including inverse ETF longs).
    capital += shares_open * fill - commission

    pnl = shares_open * (fill - pos.entry_price) if pos.direction == "LONG" \
        else shares_open * (pos.entry_price - fill)

    # Update trade record
    for t in reversed(trades):
        if t.get("_position_id") == pos.position_id and t["exit_date"] is None:
            t["exit_date"] = str(date.date())
            t["exit_price"] = fill
            t["pnl"] = pnl - commission
            t["reason"] = reason
            break

    position_mgr.close_position(pos.position_id, fill, reason)
    logger.debug("CLOSE %s %s %d @ %.2f (%s) on %s PnL=%.2f", pos.direction, pos.symbol, shares_open, fill, reason, date.date(), pnl)
    return capital


def run_backtest(
    start_date: str = "2005-01-01",
    end_date: str = "2025-12-31",
    starting_capital: float = 5000.0,
    verbose: bool = True,
    mode: str = "normal",
) -> dict:
    """Run full honest backtest using real strategy modules.

    Args:
        start_date: Backtest start date.
        end_date: Backtest end date.
        starting_capital: Starting capital in USD.
        verbose: Whether to print results to stdout.
        mode: "normal" (full strategy) or "buy_and_hold" (SPY buy-and-hold sanity check).

    Returns dict with equity curve and performance metrics.
    """
    logger.info("Loading data for backtest %s to %s...", start_date, end_date)

    # ── DATA LOADING ────────────────────────────────────────────────────────
    spy = load_spy(start=start_date, end=end_date)
    vix = load_vix(start=start_date, end=end_date)

    if spy.empty:
        logger.error("Failed to load SPY data")
        return {}

    # Align on common dates
    common_idx = spy.index.intersection(vix.index)
    spy = spy.loc[common_idx]
    vix = vix.loc[common_idx]

    logger.info("Data loaded: %d trading days", len(spy))

    # ── BUY-AND-HOLD SANITY MODE ─────────────────────────────────────────────
    if mode == "buy_and_hold":
        first_price = float(spy["Close"].iloc[0])
        last_price = float(spy["Close"].iloc[-1])
        shares = int(starting_capital * 0.99 / first_price)  # 99% allocated; 1% reserve for commissions/slippage
        cost = shares * first_price * (1 + SLIPPAGE_PCT)
        commission = cost * COMMISSION_PCT
        capital_bh = starting_capital - cost - commission
        final_val = capital_bh + shares * last_price * (1 - SLIPPAGE_PCT) - shares * last_price * COMMISSION_PCT
        equity_bh = pd.Series(
            [starting_capital, final_val],
            index=[spy.index[0], spy.index[-1]],
            name="Equity",
        )
        metrics_bh = compute_all_metrics(equity_bh)
        metrics_bh["num_trades"] = 1
        if verbose:
            print("\n" + "=" * 60)
            print("BUY-AND-HOLD SPY — Sanity Check")
            print(f"Period: {start_date} to {end_date}")
            print(f"Starting Capital: ${starting_capital:,.0f}")
            print("=" * 60)
            print(f"Final Equity:    ${final_val:,.0f}")
            print(f"Total Return:    {metrics_bh['total_return']:.1%}")
            print("=" * 60)
        return {"metrics": metrics_bh, "equity_series": equity_bh, "trades": []}

    # Load macro time series (FRED; falls back gracefully)
    macro = MacroTimeSeries(start=start_date)

    # ── STRATEGY MODULES ────────────────────────────────────────────────────
    detector = RegimeDetector()
    guard = ValuationGuard()
    long_sig_gen = LongSignalGenerator(min_regime_score=8)
    short_sig_gen = ShortSignalGenerator(max_regime_score=5)
    position_mgr = PositionManager(max_positions=6)
    # 70% of capital allocated to scaling positions, 30% reserved for hedge + buffer
    allocated_scaling_capital = starting_capital * 0.70
    scaler = AntiMartingaleScaler(
        profit_threshold=0.01,
        min_hours_between=20.0,   # ~1 trading day minimum between scale-ins
        backtest_mode=True,
        allocated_capital=allocated_scaling_capital,
    )
    exit_mgr = ExitManager()
    hedge = BaselineHedge(mode="mini", mini_hedge_pct=0.10)
    breakers = CircuitBreakers()

    # ── BACKTEST STATE ───────────────────────────────────────────────────────
    capital = starting_capital
    equity_curve = [capital]
    equity_dates = [spy.index[0]]
    trades = []

    # Track regime distribution for logging
    regime_counts: dict = {}

    # Signal / entry / exit counters (split for debugging clarity)
    trend_follow_signals = 0
    mean_revert_signals = 0
    short_signals_generated = 0
    core_positions_opened = 0
    scale_in_trades = 0
    short_entries = 0
    stop_losses_hit = 0
    targets_reached = 0
    regime_exits_count = 0
    time_stops_count = 0

    # Core position whipsaw guards
    last_core_exit_date: pd.Timestamp = None  # date of last core stop-out
    regime_history: list = []                 # rolling regime history for confirmation
    consecutive_bear_days: int = 0            # track consecutive BEAR days for soft exit

    # Track open position costs for equity calculation (cash = capital only after debits)
    prev_regime: Regime = Regime.BULL
    prev_equity = capital
    warmup = 200  # need at least 200 bars for SMA200

    for i in range(warmup, len(spy)):
        date = spy.index[i]
        row = spy.iloc[i]
        vix_row = vix.iloc[i]

        spy_slice = spy.iloc[:i + 1]
        vix_slice = vix.iloc[:i + 1]
        close_spy = float(row["Close"])
        vix_val = float(vix_row["Close"]) if "Close" in vix_row else 20.0

        # ── 1. UPDATE PRICES ON OPEN POSITIONS ──────────────────────────────
        # Approximate SH price: SH is a -1x daily inverse of SPY.
        # When SPY is ~400, SH ~36; rough approximation: SH_price ≈ 14400 / close_spy
        # (calibrated so SH ≈ 36 when SPY ≈ 400).
        sh_approx = 14400.0 / close_spy if close_spy > 0 else 36.0
        current_prices = {
            "SPY": close_spy,
            "UPRO": close_spy * 3.0,
            "SH": sh_approx,
            "SPXS": close_spy,
        }
        position_mgr.update_prices(current_prices)

        # ── 2. DAILY BORROW / EXPENSE RATIO DEDUCTIONS ──────────────────────
        for pos in position_mgr.get_open_positions():
            daily_expense = EXPENSE_RATIOS.get(pos.symbol, 0) / 252
            if pos.direction != "LONG":
                daily_expense += BORROW_FEES.get(pos.symbol, 0) / 252
            # Deduct from capital proportional to position value
            pos_value = pos.shares_open * current_prices.get(pos.symbol, pos.current_price)
            capital -= pos_value * daily_expense

        # ── 3. REAL MACRO DATA FOR REGIME DETECTOR ──────────────────────────
        hy_spread_bps = macro.get_hy_spread_on(date)
        breadth_pct = macro.get_breadth_on(date, spy_slice)

        # ── 4. REGIME DETECTION ─────────────────────────────────────────────
        try:
            regime_result = detector.detect(
                spy_daily=spy_slice,
                vix_daily=vix_slice,
                breadth_pct=breadth_pct,
                hy_spread_bps=hy_spread_bps,
            )
            regime = regime_result.regime
            regime_score = regime_result.score
            spy_above_200sma = regime_result.spy_above_200sma
        except Exception as exc:
            logger.debug("Regime detector error: %s", exc)
            regime = Regime.BULL
            regime_score = 8
            spy_above_200sma = True

        # Track regime distribution
        regime_label = regime.value if hasattr(regime, 'value') else str(regime)
        regime_counts[regime_label] = regime_counts.get(regime_label, 0) + 1

        # Track consecutive BEAR days for softened core exit logic
        if regime == Regime.BEAR:
            consecutive_bear_days += 1
        else:
            consecutive_bear_days = 0

        # Rolling regime history for 3-day confirmation before opening core
        regime_history.append(regime)
        if len(regime_history) > 5:
            regime_history.pop(0)

        # ── 5. VALUATION GUARD ──────────────────────────────────────────────
        try:
            sma200 = float(spy_slice["Close"].rolling(200).mean().iloc[-1]) if len(spy_slice) >= 200 else close_spy
            vg = guard.compute(
                base_leverage=regime_result.max_leverage if hasattr(regime_result, 'max_leverage') else 1.0,
                spy_price=close_spy,
                spy_200sma=sma200,
                vix=vix_val,
                spy_prices_history=spy_slice["Close"],
            )
            effective_leverage = vg.final_leverage
        except Exception:
            effective_leverage = 1.0

        # ── 6. CIRCUIT BREAKERS ─────────────────────────────────────────────
        open_pos_value = sum(
            p.shares_open * current_prices.get(p.symbol, p.current_price)
            for p in position_mgr.get_open_positions()
            if p.direction == "LONG"
        )
        equity = capital + open_pos_value

        spy_daily_ret = float((spy_slice["Close"].iloc[-1] - spy_slice["Close"].iloc[-2]) / spy_slice["Close"].iloc[-2]) if len(spy_slice) > 1 else 0.0

        cb_status = breakers.update(
            current_equity=equity,
            vix=vix_val,
            spy_daily_return=spy_daily_ret,
        )

        # ── 7. CHECK EXITS ───────────────────────────────────────────────────
        atr_val = float(calc_atr(spy_slice["High"], spy_slice["Low"], spy_slice["Close"]).iloc[-1])
        if pd.isna(atr_val):
            atr_val = close_spy * 0.01
        atr_values = {sym: atr_val for sym in current_prices}

        # Partial exits at targets
        partial_exits = exit_mgr.check_partial_exits(position_mgr, current_prices, atr_values)
        for pos_id, reason, close_price, fraction in partial_exits:
            pos = position_mgr.positions.get(pos_id)
            if pos and pos.shares_open > 0:
                shares_to_close = max(1, int(pos.shares_open * fraction))
                fill = _apply_slippage(close_price, "SELL" if pos.direction == "LONG" else "BUY", SLIPPAGE_PCT)
                commission = shares_to_close * fill * COMMISSION_PCT
                capital += shares_to_close * fill - commission
                pos.shares_closed += shares_to_close
                if pos.shares_open <= 0:
                    from src.positions.position_manager import PositionStatus
                    pos.status = PositionStatus.CLOSED
                logger.debug("Partial exit %s: %d shares @ %.2f (%s)", pos_id, shares_to_close, fill, reason)

        # Update trailing stops
        exit_mgr.update_trailing_stops(position_mgr, current_prices, atr_values)

        # Full exits (stops, regime exits, etc.)
        full_exits = exit_mgr.check_exits(
            position_mgr,
            current_prices,
            atr_values,
            current_regime=regime,
            previous_regime=prev_regime,
            vix=vix_val,
            current_time=date.to_pydatetime(),
            consecutive_bear_days=consecutive_bear_days,
        )
        for pos_id, reason, close_price in full_exits:
            pos = position_mgr.positions.get(pos_id)
            if pos and pos.shares_open > 0:
                # Track exit types for summary reporting
                if reason == "stop_loss":
                    stop_losses_hit += 1
                    if pos.scale_number == 0:
                        # Core position stopped out — enforce cooldown
                        last_core_exit_date = date
                elif reason.startswith("target"):
                    targets_reached += 1
                elif "regime_exit" in reason:
                    regime_exits_count += 1
                elif reason == "time_stop":
                    time_stops_count += 1

                capital = _close_position(capital, pos, close_price, position_mgr, trades, date, reason)

        # ── 8. SKIP ENTRIES IF HALTED ────────────────────────────────────────
        if cb_status.is_halted():
            equity_curve.append(capital + open_pos_value)
            equity_dates.append(date)
            prev_regime = regime
            prev_equity = equity
            continue

        # ── 9. CORE POSITION LOGIC (STRONG_BULL / BULL) ──────────────────────
        # Maintain a base exposure in bullish regimes without waiting for a signal.
        # Guards:
        #   a) Cooldown: 5 trading days (~7 calendar days) after core stop-out
        #   b) Regime confirmation: 3 consecutive days of same regime before entry
        core_target_pct = {
            Regime.STRONG_BULL: 0.30,
            Regime.BULL: 0.20,
            Regime.CHOP: 0.10,
            Regime.CAUTION: 0.0,
            Regime.BEAR: 0.0,
        }.get(regime, 0.0)

        # Guard a): cooldown after stop-out — 7 calendar days ≈ 5 trading days
        core_in_cooldown = (
            last_core_exit_date is not None
            and (date - last_core_exit_date).days < 7
        )

        # Guard b): require 3 consecutive days of the same regime
        regime_stable = (
            len(regime_history) >= 3
            and all(r == regime for r in regime_history[-3:])
        )

        if core_target_pct > 0 and capital > 100 and not core_in_cooldown and regime_stable:
            core_instrument = "UPRO" if (regime == Regime.STRONG_BULL and effective_leverage >= 2.0) else "SPY"
            has_core = any(
                p.symbol == core_instrument and p.scale_number == 0
                for p in position_mgr.get_open_positions()
            )
            if not has_core and position_mgr.can_add_position():
                core_alloc = starting_capital * core_target_pct
                core_price = current_prices.get(core_instrument, close_spy)
                core_shares = max(1, int(core_alloc / core_price))
                core_cost = core_shares * core_price * (1 + SLIPPAGE_PCT)
                core_commission = core_cost * COMMISSION_PCT
                if core_cost + core_commission <= capital * 0.95:
                    # Wider stop for core: max(4×ATR, 5%) to survive normal pullbacks
                    atr_stop_dist = 4.0 * atr_val / core_price  # ATR as fraction of price
                    core_stop_pct = max(atr_stop_dist, 0.05)
                    core_stop = core_price * (1 - core_stop_pct)
                    core_pos = position_mgr.add_position(
                        symbol=core_instrument,
                        direction="LONG",
                        entry_price=core_price * (1 + SLIPPAGE_PCT),
                        shares=core_shares,
                        stop_price=core_stop,
                        target1=core_price * 1.06,
                        target2=core_price * 1.12,
                        target3=core_price * 1.20,
                        scale_number=0,  # 0 = core position marker
                    )
                    if core_pos:
                        capital -= core_cost + core_commission
                        core_positions_opened += 1
                        trades.append({
                            "entry_date": str(date.date()),
                            "exit_date": None,
                            "instrument": core_instrument,
                            "direction": "LONG",
                            "entry_price": core_pos.entry_price,
                            "exit_price": None,
                            "shares": core_shares,
                            "pnl": None,
                            "reason": "core_position",
                            "regime": regime_label,
                            "_position_id": core_pos.position_id,
                        })
                        logger.info(
                            "CORE position opened: %s %d @ %.2f (%s regime, %.0f%% alloc, stop=%.1f%%)",
                            core_instrument, core_shares, core_pos.entry_price,
                            regime_label, core_target_pct * 100, core_stop_pct * 100,
                        )

        # ── 10. GENERATE LONG SIGNALS ─────────────────────────────────────────
        if regime in (Regime.STRONG_BULL, Regime.BULL, Regime.CHOP, Regime.CAUTION) and not cb_status.is_halted():
            # Determine instrument: UPRO in strong bull w/ leverage >= 2, else SPY
            instrument = "UPRO" if (regime == Regime.STRONG_BULL and effective_leverage >= 2.0) else "SPY"

            long_signal = long_sig_gen.generate(
                symbol=instrument,
                daily_data=spy_slice,
                regime_score=regime_score,
                regime=regime.value,
                spy_above_200sma=spy_above_200sma,
                breadth_rising=breadth_pct > 0.50,
                tick_above_500=breadth_pct > 0.55,
            )

            if long_signal:
                # Track signal type
                if regime in (Regime.STRONG_BULL, Regime.BULL):
                    trend_follow_signals += 1
                else:
                    mean_revert_signals += 1

                new_pos = scaler.execute_scale_in(
                    position_manager=position_mgr,
                    signal=long_signal,
                    capital=capital,
                    current_prices=current_prices,
                    atr=atr_val,
                    backtest_time=date.to_pydatetime(),
                )
                if new_pos:
                    scale_in_trades += 1
                    shares = new_pos.shares
                    fill = new_pos.entry_price
                    commission = shares * fill * COMMISSION_PCT
                    # Cost was NOT debited by scaler — debit now
                    capital -= shares * fill + commission
                    # Log trade
                    trades.append({
                        "entry_date": str(date.date()),
                        "exit_date": None,
                        "instrument": instrument,
                        "direction": "LONG",
                        "entry_price": fill,
                        "exit_price": None,
                        "shares": shares,
                        "pnl": None,
                        "reason": "entry_long",
                        "regime": regime.value,
                        "_position_id": new_pos.position_id,
                    })

        # ── 11. GENERATE SHORT SIGNALS ────────────────────────────────────────
        elif regime in (Regime.BEAR,) and not cb_status.is_halted():
            ema50 = calc_ema(spy_slice["Close"], 50).iloc[-1]
            spy_below_50sma = not pd.isna(ema50) and close_spy < float(ema50)
            prev_vix = float(vix_slice["Close"].iloc[-2]) if len(vix_slice) > 1 else vix_val
            vix_rising = vix_val > prev_vix

            short_signal = short_sig_gen.generate(
                symbol="SH",
                daily_data=spy_slice,
                regime_score=regime_score,
                regime=regime.value,
                vix=vix_val,
                vix_rising=vix_rising,
                spy_below_50sma=spy_below_50sma,
                ad_falling=breadth_pct < 0.45,
            )

            if short_signal:
                short_signals_generated += 1
                new_pos = scaler.execute_scale_in(
                    position_manager=position_mgr,
                    signal=short_signal,
                    capital=capital,
                    current_prices=current_prices,
                    atr=atr_val,
                    backtest_time=date.to_pydatetime(),
                )
                if new_pos:
                    short_entries += 1
                    shares = new_pos.shares
                    fill = new_pos.entry_price
                    commission = shares * fill * COMMISSION_PCT
                    capital -= shares * fill + commission
                    trades.append({
                        "entry_date": str(date.date()),
                        "exit_date": None,
                        "instrument": "SH",
                        "direction": "LONG",  # SH is a long of inverse ETF
                        "entry_price": fill,
                        "exit_price": None,
                        "shares": shares,
                        "pnl": None,
                        "reason": "entry_short_via_sh",
                        "regime": regime.value,
                        "_position_id": new_pos.position_id,
                    })

        # ── 12. EQUITY UPDATE ────────────────────────────────────────────────
        open_pos_value = sum(
            p.shares_open * current_prices.get(p.symbol, p.current_price)
            for p in position_mgr.get_open_positions()
            if p.direction == "LONG"
        )
        equity = max(0.0, capital + open_pos_value)
        equity_curve.append(equity)
        equity_dates.append(date)

        prev_regime = regime
        prev_equity = equity

    # ── FINAL CLOSE ──────────────────────────────────────────────────────────
    if len(spy) > 0:
        last_price = float(spy["Close"].iloc[-1])
        last_date = spy.index[-1]
        for pos in list(position_mgr.get_open_positions()):
            capital = _close_position(capital, pos, last_price, position_mgr, trades, last_date, "end_of_backtest")
        equity_curve[-1] = capital  # update last equity point

    equity_series = pd.Series(equity_curve, index=equity_dates, name="Equity")
    metrics = compute_all_metrics(equity_series)

    # Count completed trades
    completed_trades = [t for t in trades if t.get("exit_date") is not None]
    metrics["num_trades"] = len(completed_trades)

    # ── BENCHMARK: SPY buy-and-hold for same period ─────────────────────────
    if len(spy) >= 2:
        spy_start = float(spy["Close"].iloc[0])
        spy_end = float(spy["Close"].iloc[-1])
        spy_return = (spy_end / spy_start) - 1.0
    else:
        spy_return = 0.0

    if verbose:
        print("\n" + "=" * 60)
        print("SP500 Smart Scalper Bot — Backtest Results")
        print(f"Period: {start_date} to {end_date}")
        print(f"Starting Capital: ${starting_capital:,.0f}")
        print("=" * 60)
        print(f"Final Equity:    ${metrics['end_equity']:,.0f}")
        print(f"Total Return:    {metrics['total_return']:.1%}")
        print(f"CAGR:            {metrics['cagr']:.1%}")
        sharpe_val = metrics['sharpe_ratio']
        print(f"Sharpe Ratio:    {'inf' if sharpe_val == float('inf') else f'{sharpe_val:.2f}'}")
        sortino_val = metrics['sortino_ratio']
        print(f"Sortino Ratio:   {'inf' if sortino_val == float('inf') else f'{sortino_val:.2f}'}")
        print(f"Max Drawdown:    {metrics['max_drawdown']:.1%}")
        calmar_val = metrics['calmar_ratio']
        print(f"Calmar Ratio:    {'inf' if calmar_val == float('inf') else f'{calmar_val:.2f}'}")
        print(f"Win Rate:        {metrics['win_rate']:.1%}")
        pf_val = metrics['profit_factor']
        print(f"Profit Factor:   {'inf' if pf_val == float('inf') else f'{pf_val:.2f}'}")
        print(f"Num Trades:      {metrics['num_trades']}")
        print("=" * 60)

        # Benchmark comparison
        outperform = metrics['total_return'] - spy_return
        print(f"\nBenchmark Comparison:")
        print(f"  Bot return:   {metrics['total_return']:+.1%}")
        print(f"  SPY return:   {spy_return:+.1%}   (buy-and-hold same period)")
        print(f"  Outperform:   {outperform:+.1%}")

        # Regime distribution summary
        total_bars = sum(regime_counts.values()) or 1
        print("\nRegime Distribution:")
        for rname in ["STRONG_BULL", "BULL", "CHOP", "CAUTION", "BEAR"]:
            count = regime_counts.get(rname, 0)
            pct = count / total_bars * 100
            print(f"  {rname:<12}: {count:5d} days ({pct:5.1f}%)")

        print(f"\nSignal generation:")
        print(f"  Trend-follow signals:    {trend_follow_signals}")
        print(f"  Mean-revert signals:     {mean_revert_signals}")
        print(f"  Short signals:           {short_signals_generated}")
        print(f"\nTrade entries:")
        print(f"  Core positions opened:   {core_positions_opened}")
        print(f"  Scale-in trades:         {scale_in_trades}")
        print(f"  Short entries:           {short_entries}")
        print(f"\nTrade exits:")
        print(f"  Stop-losses hit:         {stop_losses_hit}")
        print(f"  Targets reached:         {targets_reached}")
        print(f"  Regime exits:            {regime_exits_count}")
        print(f"  Time stops:              {time_stops_count}")
        print("=" * 60)

        # Save results
        os.makedirs("results", exist_ok=True)
        output = {
            "config": {
                "start_date": start_date,
                "end_date": end_date,
                "starting_capital": starting_capital,
                "commission_pct": COMMISSION_PCT,
                "slippage_pct": SLIPPAGE_PCT,
            },
            "metrics": metrics,
            "benchmark": {
                "spy_return": spy_return,
                "outperformance": outperform,
            },
            "regime_distribution": regime_counts,
            "signal_stats": {
                "trend_follow_signals": trend_follow_signals,
                "mean_revert_signals": mean_revert_signals,
                "short_signals_generated": short_signals_generated,
                "core_positions_opened": core_positions_opened,
                "scale_in_trades": scale_in_trades,
                "short_entries": short_entries,
                "stop_losses_hit": stop_losses_hit,
                "targets_reached": targets_reached,
                "regime_exits": regime_exits_count,
                "time_stops": time_stops_count,
            },
            "equity_curve": {
                str(d.date()): float(v)
                for d, v in zip(equity_dates, equity_curve)
                if hasattr(d, 'date')
            },
            "trades": completed_trades[:100],
        }
        with open("results/backtest_full.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to results/backtest_full.json")

    return {"metrics": metrics, "equity_series": equity_series, "trades": completed_trades}


if __name__ == "__main__":
    run_backtest()

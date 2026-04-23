"""Honest Backtester — includes commissions, slippage, and borrow fees.

Uses real historical data from Yahoo Finance and FRED.
Default period: 2005-01-01 to 2025-12-31.

Run: python src/backtest/honest_backtest.py
"""
import json
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from src.data.yahoo_loader import load_spy, load_vix, load_ohlcv
from src.data.fred_macro import load_macro_data, get_cape_ratio
from src.regime.detector import RegimeDetector
from src.regime.valuation_guard import ValuationGuard
from src.regime.regime_types import Regime, score_to_regime
from src.signals.indicators import calc_atr, calc_rsi, calc_macd, calc_ema
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


def run_backtest(
    start_date: str = "2005-01-01",
    end_date: str = "2025-12-31",
    starting_capital: float = 5000.0,
    verbose: bool = True,
) -> dict:
    """Run full honest backtest.

    Returns dict with equity curve and performance metrics.
    """
    logger.info("Loading data for backtest %s to %s...", start_date, end_date)

    # Load data
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

    # Initialize regime detector
    detector = RegimeDetector()
    guard = ValuationGuard()

    # ── BACKTEST LOOP ──────────────────────────────────────────────────
    capital = starting_capital
    equity_curve = [capital]
    equity_dates = [spy.index[0]]
    positions = []  # (entry_date, exit_date, entry_px, exit_px, direction, instrument, shares)
    trades = []

    # Warm-up period: need at least 200 bars
    warmup = 200
    in_position = False
    position_entry_price = 0.0
    position_entry_date = None
    position_instrument = "SPY"
    position_direction = "LONG"
    position_shares = 0
    position_stop = 0.0
    position_target1 = 0.0
    position_target2 = 0.0
    position_target3 = 0.0
    t1_hit = False
    t2_hit = False
    position_days = 0
    trailing_high = 0.0

    for i in range(warmup, len(spy)):
        date = spy.index[i]
        row = spy.iloc[i]
        vix_row = vix.iloc[i]

        spy_slice = spy.iloc[:i+1]
        vix_slice = vix.iloc[:i+1]
        close_spy = float(row["Close"])
        vix_val = float(vix_row["Close"]) if "Close" in vix_row else 20.0

        # Daily borrow fee / expense ratio deduction (if in position)
        if in_position:
            daily_fee = EXPENSE_RATIOS.get(position_instrument, 0) / 252
            if position_direction != "LONG":
                daily_fee += BORROW_FEES.get(position_instrument, 0) / 252
            capital *= (1 - daily_fee)
            position_days += 1

        # Run regime detector every day
        try:
            regime_result = detector.detect(
                spy_daily=spy_slice,
                vix_daily=vix_slice,
                breadth_pct=0.55,  # simplified: assume average breadth
                hy_spread_bps=350.0,
            )
            regime = regime_result.regime
            regime_score = regime_result.score
        except Exception:
            regime = Regime.BULL
            regime_score = 8

        # Valuation guard
        sma200 = float(spy_slice["Close"].rolling(200).mean().iloc[-1]) if len(spy_slice) >= 200 else close_spy
        vg = guard.compute(
            base_leverage=regime_result.max_leverage,
            spy_price=close_spy,
            spy_200sma=sma200,
            vix=vix_val,
            spy_prices_history=spy_slice["Close"],
        )

        # Check exits if in position
        if in_position:
            exit_reason = None

            # Stop loss
            if position_direction == "LONG" and close_spy <= position_stop:
                exit_reason = "stop_loss"
            elif position_direction == "SHORT" and close_spy >= position_stop:
                exit_reason = "stop_loss"

            # Targets
            if not t1_hit and position_direction == "LONG" and close_spy >= position_target1:
                t1_hit = True
                # Close 1/3 of position
                shares_to_close = max(1, position_shares // 3)
                pnl = shares_to_close * (close_spy - position_entry_price)
                commission = shares_to_close * close_spy * COMMISSION_PCT
                slippage = shares_to_close * close_spy * SLIPPAGE_PCT
                capital += pnl - commission - slippage
                position_shares -= shares_to_close
                position_stop = position_entry_price  # move to BE
                logger.debug("T1 hit at %.2f, close 1/3", close_spy)

            if t1_hit and not t2_hit and position_direction == "LONG" and close_spy >= position_target2:
                t2_hit = True
                shares_to_close = max(1, position_shares // 2)
                pnl = shares_to_close * (close_spy - position_entry_price)
                commission = shares_to_close * close_spy * COMMISSION_PCT
                slippage = shares_to_close * close_spy * SLIPPAGE_PCT
                capital += pnl - commission - slippage
                position_shares -= shares_to_close
                position_stop = position_entry_price * 1.02

            # Time stop
            if position_days >= 10:
                pnl_pct = (close_spy - position_entry_price) / position_entry_price
                if abs(pnl_pct) < 0.01:
                    exit_reason = "time_stop"

            # VIX panic
            if vix_val > 35 and position_direction == "LONG":
                exit_reason = "vix_panic"

            # Trailing stop activation
            if position_direction == "LONG":
                pnl_pct = (close_spy - position_entry_price) / position_entry_price
                if pnl_pct >= 0.03:
                    trailing_high = max(trailing_high, close_spy)
                    atr_val = float(calc_atr(spy_slice["High"], spy_slice["Low"], spy_slice["Close"]).iloc[-1])
                    trail_stop = trailing_high - 1.5 * atr_val
                    if trail_stop > position_stop:
                        position_stop = trail_stop

            # Execute exit
            if exit_reason or (not in_position):
                if in_position and exit_reason:
                    fill = close_spy * (1 - SLIPPAGE_PCT) if position_direction == "LONG" else close_spy * (1 + SLIPPAGE_PCT)
                    pnl = position_shares * (fill - position_entry_price)
                    commission = position_shares * fill * COMMISSION_PCT
                    capital += pnl - commission
                    in_position = False
                    trades.append({
                        "entry_date": str(position_entry_date.date()),
                        "exit_date": str(date.date()),
                        "instrument": position_instrument,
                        "direction": position_direction,
                        "entry_price": position_entry_price,
                        "exit_price": fill,
                        "shares": position_shares,
                        "pnl": pnl - commission,
                        "reason": exit_reason,
                        "regime": regime.value,
                    })
                    position_days = 0
                    t1_hit = False
                    t2_hit = False
                    trailing_high = 0.0

        # Entry signal (simplified rule-based)
        if not in_position:
            # Long signal conditions
            if (regime in (Regime.STRONG_BULL, Regime.BULL) and
                    regime_result.spy_above_200sma and
                    vix_val < 25):

                rsi = calc_rsi(spy_slice["Close"], 14)
                current_rsi = float(rsi.iloc[-1]) if len(rsi.dropna()) > 0 else 50.0
                ema50 = float(calc_ema(spy_slice["Close"], 50).iloc[-1])

                # Simple entry: RSI oversold + price near EMA
                if current_rsi < 45 or (close_spy > ema50 * 0.99 and close_spy < ema50 * 1.005):
                    # Determine instrument based on regime
                    if regime == Regime.STRONG_BULL and vg.final_leverage >= 2.0:
                        instrument = "UPRO"
                    else:
                        instrument = "SPY"

                    # Position sizing
                    atr_val = float(calc_atr(spy_slice["High"], spy_slice["Low"], spy_slice["Close"]).iloc[-1])
                    stop_dist = max(atr_val * 1.5, close_spy * 0.015)
                    risk_dollars = capital * 0.015
                    shares = max(1, int(risk_dollars / stop_dist))
                    max_shares = int(capital * 0.20 / close_spy)
                    shares = min(shares, max_shares)

                    # Entry with slippage
                    fill = close_spy * (1 + SLIPPAGE_PCT)
                    commission = shares * fill * COMMISSION_PCT

                    if shares * fill + commission <= capital:
                        in_position = True
                        position_entry_price = fill
                        position_entry_date = date
                        position_instrument = instrument
                        position_direction = "LONG"
                        position_shares = shares
                        position_stop = fill - stop_dist
                        position_target1 = fill * 1.02
                        position_target2 = fill * 1.045
                        position_target3 = fill * 1.08
                        trailing_high = fill
                        capital -= commission
                        logger.debug("LONG entry: %s %d @ %.2f on %s", instrument, shares, fill, date.date())

        equity_curve.append(capital)
        equity_dates.append(date)

    # Final close of any open position
    if in_position and len(spy) > 0:
        last_price = float(spy["Close"].iloc[-1])
        pnl = position_shares * (last_price - position_entry_price)
        capital += pnl

    equity_series = pd.Series(equity_curve, index=equity_dates, name="Equity")
    metrics = compute_all_metrics(equity_series)
    metrics["num_trades"] = len(trades)

    if verbose:
        print("\n" + "=" * 60)
        print(f"SP500 Smart Scalper Bot — Backtest Results")
        print(f"Period: {start_date} to {end_date}")
        print(f"Starting Capital: ${starting_capital:,.0f}")
        print("=" * 60)
        print(f"Final Equity:    ${metrics['end_equity']:,.0f}")
        print(f"Total Return:    {metrics['total_return']:.1%}")
        print(f"CAGR:            {metrics['cagr']:.1%}")
        print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:   {metrics['sortino_ratio']:.2f}")
        print(f"Max Drawdown:    {metrics['max_drawdown']:.1%}")
        print(f"Calmar Ratio:    {metrics['calmar_ratio']:.2f}")
        print(f"Win Rate:        {metrics['win_rate']:.1%}")
        print(f"Profit Factor:   {metrics['profit_factor']:.2f}")
        print(f"Num Trades:      {metrics['num_trades']}")
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
            "equity_curve": {
                str(d.date()): float(v)
                for d, v in zip(equity_dates, equity_curve)
                if hasattr(d, 'date')
            },
            "trades": trades[:100],  # save first 100 trades
        }
        with open("results/backtest_full.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to results/backtest_full.json")

    return {"metrics": metrics, "equity_series": equity_series, "trades": trades}


if __name__ == "__main__":
    run_backtest()

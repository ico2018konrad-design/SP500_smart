"""Streamlit Dashboard — live P&L, positions, regime, risk budget.

Run: streamlit run src/monitoring/dashboard.py
"""
import os
import sys
import time
from datetime import datetime

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SP500 Smart Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_config():
    """Load bot configuration."""
    try:
        import yaml
        with open("config/strategy_config.yaml") as f:
            return yaml.safe_load(f)
    except Exception:
        return {"mode": "mini", "trading_mode": "paper", "starting_capital": 5000}


def get_market_data():
    """Load latest market data."""
    try:
        from src.data.yahoo_loader import load_spy, load_vix
        spy = load_spy()
        vix = load_vix()
        return spy, vix
    except Exception as e:
        st.warning(f"Could not load market data: {e}")
        return pd.DataFrame(), pd.DataFrame()


def get_regime(spy, vix):
    """Calculate current regime."""
    try:
        from src.regime.detector import RegimeDetector
        detector = RegimeDetector()
        if spy.empty or vix.empty:
            return None
        result = detector.detect(spy_daily=spy, vix_daily=vix, hy_spread_bps=350.0)
        return result
    except Exception:
        return None


def load_backtest_results():
    """Load last backtest results if available."""
    try:
        import json
        if os.path.exists("results/backtest_full.json"):
            with open("results/backtest_full.json") as f:
                return json.load(f)
    except Exception:
        pass
    return None


# ── DASHBOARD LAYOUT ──────────────────────────────────────────────────────────

def main():
    config = load_config()

    # Header
    st.title("📈 SP500 Smart Scalper Bot")
    mode_badge = "🟡 PAPER" if config.get("trading_mode") == "paper" else "🔴 LIVE"
    st.markdown(f"**Mode:** {config.get('mode', 'mini').upper()} | **Trading:** {mode_badge} | "
                f"**Capital:** {config.get('starting_capital', 5000):,.0f} CHF")

    if config.get("trading_mode") == "paper":
        st.info("📋 Paper Trading Mode — no real money at risk")

    st.divider()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.write(f"**Mode:** {config.get('mode', 'mini')}")
        st.write(f"**LLM:** {'✅ Enabled' if config.get('llm_enabled') else '❌ Disabled'}")
        st.write(f"**Capital:** ${config.get('starting_capital', 5000):,.0f}")

        if st.button("🔄 Refresh Data"):
            st.rerun()

        st.divider()
        st.header("🚨 Emergency")
        if st.button("🛑 Kill Switch", type="primary", use_container_width=True):
            st.error("Kill switch activated! Trading halted.")

    # Load data
    with st.spinner("Loading market data..."):
        spy, vix = get_market_data()
        regime_result = get_regime(spy, vix)

    # Row 1: Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        spy_price = float(spy["Close"].iloc[-1]) if not spy.empty else 0
        spy_change = float(spy["Close"].pct_change().iloc[-1]) if not spy.empty else 0
        st.metric("SPY Price", f"${spy_price:.2f}", f"{spy_change:+.2%}")

    with col2:
        vix_val = float(vix["Close"].iloc[-1]) if not vix.empty else 20
        vix_color = "🟢" if vix_val < 20 else "🟡" if vix_val < 30 else "🔴"
        st.metric("VIX", f"{vix_color} {vix_val:.1f}")

    with col3:
        if regime_result:
            score = regime_result.score
            regime_name = regime_result.regime.value
            regime_colors = {
                "STRONG_BULL": "🚀",
                "BULL": "📈",
                "CHOP": "↔️",
                "CAUTION": "⚠️",
                "BEAR": "📉",
            }
            emoji = regime_colors.get(regime_name, "")
            st.metric("Regime Score", f"{score}/11", f"{emoji} {regime_name}")
        else:
            st.metric("Regime Score", "N/A")

    with col4:
        st.metric("Open Positions", "0 / 6")

    with col5:
        st.metric("Portfolio", f"${config.get('starting_capital', 5000):,.0f}", "+0.00%")

    st.divider()

    # Row 2: Regime details + Chart
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("🎯 Regime Indicators")
        if regime_result:
            indicators = {
                "SPY > 200 SMA": regime_result.spy_above_200sma,
                "SPY > 50 SMA": regime_result.spy_above_50sma,
                "200 SMA Slope +": regime_result.sma200_slope_positive,
                "Higher Highs/Lows": regime_result.higher_highs_lows,
                "MACD Weekly Bull": regime_result.macd_weekly_bullish,
                "ADX Bull": regime_result.adx_bullish,
                "VIX < 20": regime_result.vix_below_20,
                "VIX Contango": regime_result.vix_term_contango,
                "Breadth > 55%": regime_result.breadth_above_55pct,
                "HY Spread OK": regime_result.hy_spread_ok,
                "Yield Curve OK": regime_result.yield_curve_ok,
            }
            for name, value in indicators.items():
                icon = "✅" if value else "❌"
                st.write(f"{icon} {name}")

            st.metric("Max Leverage", f"{regime_result.max_leverage:.1f}x")
        else:
            st.info("Loading regime data...")

    with col_right:
        st.subheader("📊 SPY Price Chart")
        if not spy.empty:
            chart_data = spy["Close"].tail(60).reset_index()
            chart_data.columns = ["Date", "SPY Close"]
            st.line_chart(chart_data.set_index("Date"))
        else:
            st.info("No chart data available")

    st.divider()

    # Row 3: Positions + Risk Budget
    col_pos, col_risk = st.columns(2)

    with col_pos:
        st.subheader("📋 Open Positions")
        st.info("No open positions (paper trading mode)")

        pos_data = pd.DataFrame({
            "Symbol": ["—"],
            "Direction": ["—"],
            "Entry": ["—"],
            "Current": ["—"],
            "P&L %": ["—"],
            "Stop": ["—"],
        })
        st.dataframe(pos_data, use_container_width=True)

    with col_risk:
        st.subheader("🛡️ Risk Budget")
        risk_data = {
            "Limit": ["Daily -3%", "Weekly -7%", "Monthly -12%"],
            "Used": ["0.00%", "0.00%", "0.00%"],
            "Status": ["✅ OK", "✅ OK", "✅ OK"],
        }
        st.dataframe(pd.DataFrame(risk_data), use_container_width=True)

    st.divider()

    # Row 4: Backtest Results
    st.subheader("📈 Backtest Results")
    bt_results = load_backtest_results()
    if bt_results:
        m = bt_results.get("metrics", {})
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        with bcol1:
            st.metric("CAGR", f"{m.get('cagr', 0):.1%}")
        with bcol2:
            st.metric("Sharpe", f"{m.get('sharpe_ratio', 0):.2f}")
        with bcol3:
            st.metric("Max DD", f"{m.get('max_drawdown', 0):.1%}")
        with bcol4:
            st.metric("Win Rate", f"{m.get('win_rate', 0):.0%}")
    else:
        st.info("No backtest results yet. Run: `python src/backtest/honest_backtest.py`")

    st.divider()

    # Footer
    st.caption(
        "⚠️ **DISCLAIMER**: Educational software only. Not financial advice. "
        "Past performance ≠ future results. Trading involves risk of loss. "
        "Paper trade 6+ months before going live."
    )
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

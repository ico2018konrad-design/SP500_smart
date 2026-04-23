"""SP500 Smart Scalper Bot — Main Entry Point.

Default: paper trading, mini mode.
Run: python src/main.py

IMPORTANT: This defaults to paper trading.
To enable live trading, set trading_mode: live in config
AND TRADING_MODE=live in .env (requires explicit opt-in).
"""
import logging
import os
import signal
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/bot_{datetime.now().strftime('%Y%m%d')}.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Main bot entry point."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    logger.info("=" * 60)
    logger.info("SP500 Smart Scalper Bot v1.0")
    logger.info("Starting at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S CET"))

    # Load configuration
    try:
        import yaml
        with open("config/strategy_config.yaml") as f:
            config = yaml.safe_load(f)
        mode = config.get("mode", "mini")
        trading_mode = config.get("trading_mode", "paper")
        llm_enabled = config.get("llm_enabled", False)
        starting_capital = config.get("starting_capital", 5000.0)
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        mode = "mini"
        trading_mode = "paper"
        llm_enabled = False
        starting_capital = 5000.0

    logger.info("Mode: %s | Trading: %s | LLM: %s | Capital: %.0f CHF",
                mode, trading_mode, llm_enabled, starting_capital)

    # Safety check
    if trading_mode == "live":
        logger.warning("=" * 60)
        logger.warning("⚠️  LIVE TRADING MODE — REAL MONEY AT RISK  ⚠️")
        logger.warning("=" * 60)
        response = input("Type 'CONFIRM LIVE TRADING' to proceed: ")
        if response != "CONFIRM LIVE TRADING":
            logger.info("Live trading aborted by user")
            sys.exit(0)

    # Initialize components
    from src.execution.paper_trader import PaperTrader
    from src.regime.detector import RegimeDetector
    from src.regime.valuation_guard import ValuationGuard
    from src.signals.long_signals import LongSignalGenerator
    from src.signals.short_signals import ShortSignalGenerator
    from src.positions.position_manager import PositionManager
    from src.positions.anti_martingale_scaler import AntiMartingaleScaler
    from src.positions.exit_manager import ExitManager
    from src.hedge.baseline_hedge import BaselineHedge
    from src.hedge.reactive_hedge import ReactiveHedge
    from src.hedge.panic_hedge import PanicHedge
    from src.risk.circuit_breakers import CircuitBreakers
    from src.risk.kill_switch import KillSwitch
    from src.schedule.daily_scheduler import DailyScheduler

    paper_trader = PaperTrader(starting_capital=starting_capital)
    regime_detector = RegimeDetector()
    valuation_guard = ValuationGuard()
    long_signals = LongSignalGenerator()
    short_signals = ShortSignalGenerator()
    position_manager = PositionManager()
    scaler = AntiMartingaleScaler()
    exit_manager = ExitManager()
    baseline_hedge = BaselineHedge(mode=mode)
    reactive_hedge = ReactiveHedge()
    panic_hedge = PanicHedge()
    circuit_breakers = CircuitBreakers()
    kill_switch = KillSwitch()
    scheduler = DailyScheduler(paper_trader=paper_trader, regime_detector=regime_detector)

    # Connect to IBKR if available
    paper_trader.connect_ibkr()

    # Setup graceful shutdown
    def shutdown(sig, frame):
        logger.info("Shutdown signal received — closing positions...")
        kill_switch.activate("Graceful shutdown", activated_by="auto", requires_manual_reset=False)
        scheduler.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("Bot initialized successfully. Starting scheduler...")

    # Run morning briefing immediately on start
    scheduler.morning_briefing()

    # Start scheduler
    scheduler.setup_schedule()

    # Main loop
    logger.info("Entering main loop (5-minute scan interval)...")
    last_scan = datetime.now()

    while True:
        # Kill switch check
        if kill_switch.is_active():
            logger.warning("Kill switch active — trading halted")
            time.sleep(60)
            continue

        # Circuit breaker check
        equity = paper_trader.get_equity()
        cb_status = circuit_breakers.update(equity)
        if not circuit_breakers.can_trade():
            logger.info(
                "Circuit breaker active: %s | Halt until: %s",
                cb_status.halt_reason, cb_status.halt_until
            )
            time.sleep(60)
            import schedule as sched
            sched.run_pending()
            continue

        # Blackout check
        if scheduler.is_blackout:
            time.sleep(30)
            import schedule as sched
            sched.run_pending()
            continue

        now = datetime.now()
        elapsed = (now - last_scan).total_seconds()

        if elapsed >= 300:  # 5-minute scan
            last_scan = now
            logger.debug("Running 5-minute scan...")

            try:
                from src.data.yahoo_loader import load_spy, load_vix
                spy = load_spy()
                vix = load_vix()

                if not spy.empty and not vix.empty:
                    from src.data.fred_macro import get_current_hy_spread
                    hy_spread = get_current_hy_spread()
                    regime_result = regime_detector.detect(
                        spy_daily=spy,
                        vix_daily=vix,
                        hy_spread_bps=hy_spread,
                    )
                    logger.debug(
                        "Regime: %s (%d/11)", regime_result.regime.value, regime_result.score
                    )

            except Exception as exc:
                logger.error("Scan error: %s", exc)

        time.sleep(10)
        import schedule as sched
        sched.run_pending()


if __name__ == "__main__":
    main()

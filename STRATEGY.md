# SP500 Smart Scalper Bot — Full Strategy Specification

## 1. Philosophy & Targets

- **Long bias**: market uptrends 75% of time, short only in confirmed bear regime
- **Hedge ALWAYS active**: baseline insurance at all times
- **Leverage only** when multiple indicators confirm BULL
- **Survival > max profit**: capital preservation is primary
- **Targets**: CAGR 25-35%, stretch 60%+ in bull years, Max DD -12%, Sharpe 1.5+

## 2. Architecture (5 Layers)

```
1. Regime Detector     (11 indicators → score 0-11)
2. Signal Engine       (LONG/SHORT entries: Setup+Trigger+Confirm)
3. Position Manager    (max 6 scaled positions, anti-martingale)
4. Hedge Module        (3 levels: baseline, reactive, panic)
5. Risk Guard          (circuit breakers, kill switches)
```

## 3. Instruments & Allocation

| Role | Instrument | When |
|------|------------|------|
| Core long | SPY | Always |
| Leveraged long | UPRO (3x) | STRONG BULL only |
| Short exposure | SH (-1x) or SPXS (-3x) | Chop/Bear |
| Hedge Phase 2 | SPY put spreads | Capital >= 15k CHF |
| Panic hedge | VXX | VIX spike events |

## 4. Regime Detector (11 indicators)

**TREND (4 pts)**
- SPY > 200 SMA
- SPY > 50 SMA
- 200 SMA slope positive (30d)
- Higher highs + higher lows structure

**MOMENTUM (2 pts)**
- Weekly MACD bullish
- ADX(14) > 25 with DI+ > DI-

**VOLATILITY (2 pts)**
- VIX < 20
- VIX term structure in contango (VIX9D < VIX < VX1)

**BREADTH (1 pt)**
- % of S&P 500 stocks above 50 SMA > 55%

**CREDIT/MACRO (2 pts)**
- HY credit spreads < 400 bps
- Yield curve not freshly inverted

Score → Regime:
- 10-11: STRONG BULL (max leverage 2.5x)
- 8-9: BULL (2.0x)
- 6-7: CHOP (1.0x)
- 4-5: CAUTION (0.5x)
- 0-3: BEAR (0x, short bias)

## 5. Valuation Guard (KEY INNOVATION)

```python
penalty = 1.0
if spy_price > spy_200_sma * 1.15: penalty *= 0.75  # overextension
if cape_ratio > 35:                 penalty *= 0.75  # high valuation
if vix < 13:                        penalty *= 0.80  # complacency
if months_since_last_10pct_correction > 18: penalty *= 0.70
final_leverage = base_leverage * penalty
```

## 6. LONG Entry Signals — ADAPTIVE MODE (v1.2)

### Signal Mode Selection (automatic, based on regime)

```
STRONG_BULL / BULL  →  TREND-FOLLOW mode   (momentum/continuation entries)
CHOP / CAUTION      →  MEAN-REVERT mode    (oversold dip-buying entries)
BEAR                →  No longs (shorts handled separately)
```

**Level 1 — SETUP (ALL required)**
- Regime score >= 8 (CHOP/CAUTION: >= 6)
- SPY > 200 SMA
- No major event in 24h (FOMC, CPI, NFP)
- ATR(14) < 3% (not chaotic)
- No opposing SHORT position open

---

### TREND-FOLLOW Triggers (STRONG_BULL / BULL)

**Level 2 — TRIGGER (min 2 of 5)**
1. Pullback to 20 EMA in uptrend: `close between 20EMA×0.99 and 20EMA×1.02 AND 20EMA > 50EMA`
2. Breakout above 20-day high: `close > max(close[-20:-1]) AND green candle`
3. Higher-low pattern: recent swing low > previous swing low in 20-bar window, currently bouncing
4. Momentum confirmation: 5d return > 1% AND RSI(14) in 50-70 range (healthy, not exhausted)
5. Golden cross / above both EMAs: `50EMA > 200EMA AND close > 50EMA AND close > 200EMA`

**Level 3 — CONFIRMATION (min 1 of 3, relaxed)**
1. Volume > 1.1x 20-bar avg
2. Higher close than previous 3 days
3. Breadth indicator rising

---

### MEAN-REVERT Triggers (CHOP / CAUTION)

**Level 2 — TRIGGER (min 2 of 5 daily, min 3 of 5 intraday)**

Thresholds auto-adjusted for timeframe:
- **Daily**: RSI threshold 45, Stochastic threshold 35
- **Intraday**: RSI threshold 40, Stochastic threshold 30

1. RSI below threshold, crossing up
2. Price touches 50 EMA and bounces
3. MACD histogram rising from negative
4. Stochastic below threshold, bullish cross (%K > %D)
5. Bollinger Band lower touch + green candle

**Level 3 — CONFIRMATION (min 2 of 4)**
1. Volume > 1.3x 20-bar avg
2. Price above VWAP
3. A/D line rising (proxy)
4. TICK index > +500 NYSE (proxy)

**Asymmetric R:R**: Stop -1.5%, T1 +2%, T2 +4.5%, T3 +8%. Min R:R 1:2.0.

---

### Timeframe Detection

The signal generator auto-detects whether data is **intraday** (bar gap < 1 day) or **daily** (gap >= 1 day) and adjusts trigger thresholds accordingly. This ensures correct behavior whether running on live 1H bars or historical daily bars.

---

### Core Position Concept (v1.2)

In bullish regimes, the bot maintains a **base exposure** without waiting for a trigger signal:

| Regime | Core % | Instrument |
|--------|--------|------------|
| STRONG_BULL (leverage>=2) | 30% | UPRO |
| STRONG_BULL (leverage<2) | 30% | SPY |
| BULL | 20% | SPY |
| CHOP | 10% | SPY |
| CAUTION/BEAR | 0% | — |

Core positions use scale_number=0 (distinguished from scaled positions 1-6) and wider stops (3×ATR).


## 7. SHORT Entry Signals

**SETUP**: Regime score <= 5, VIX > 20 rising, SPY < 50 SMA, no panic rebound.

**TRIGGER (min 3 of 5)**: RSI > 68 crossing down, price rejected from 50 EMA, MACD bearish cross, Stoch > 75 bearish cross, BB upper rejection.

**CONFIRMATION (min 2 of 4)**: Red volume > green (3-bar), VIX rising >5% intraday, Put/Call > 1.0, A/D falling.

Execute SHORT via SPXS or SH — NOT naked short. Size 1/3 of normal long. R:R min 1:1.5.

## 8. Scale-in Logic (Anti-Martingale)

Max 6 positions, each 16.67% of **allocated capital** (70% of starting capital reserved for scaled positions; 30% for hedge + cash buffer).

**Equal-dollar sizing**: Position size is calculated from the initial allocated capital, not current cash. This ensures Position #6 is the same dollar size as Position #1, not degraded by prior trades.

**New position ONLY when previous is +1% in profit** (never averaging down).

```
Entry:    Pos #1 opened at $100
+1%:      $101 → open Pos #2, move Pos #1 stop to break-even
+1%:      Pos #2 at +1% → open Pos #3, trail stops up
Continue up to 6 positions
```

Rules:
- NEVER add to losing position
- Min 1-2 hours between entries
- Each position has own stop + 3 targets
- After T1: stop moves to BE. After T2: stop to +2%

## 9. Exit Logic

**A) Profit Ladder**: T1 (+2%) close 1/3, T2 (+4.5%) close 1/3, T3 (+8%) close last 1/3.

**B) Trailing Stop**: After +3% gain: Stop = HH - 1.5×ATR(14).

**C) Regime-based Forced Exit**:
- BULL→CHOP: close 50% UPRO immediately
- CHOP→BEAR: flat all longs, switch to shorts
- VIX > 30 with open positions: close within 1 hour
- SPY -2% in 1 hour: FLAT ALL

**D) Time Stop**: Open > 10 days without ±1% movement → close.

## 10. Hedge System

**Level 1 — Baseline** (always active):
- Phase 1 (<15k): 10% in SH (-1x)
- Phase 2 (>=15k): SPY put spread @ 93%/82%, 90 DTE, 5-7%/month, rolled every 60 days

**Level 2 — Reactive** (dynamic):
- SPY -1% AND VIX +10% simultaneously → buy SPXS 10% capital
- Regime downgrade BULL→CHOP → increase SH from 10% to 20%

**Level 3 — Panic**:
- Triggers: VIX > 35, SPY -5% in 2 sessions, HY +100bps/week, LLM detects systemic event
- Actions: flat all longs, buy VXX 3%, SPXS 25%, Telegram alert, manual unlock required
- Min 5 trading days in panic mode

## 11. Risk Management

**Per Trade**: Max 1.5% risk (dynamic 0.5-2% by regime), hard stop required.

**Per Day**: -3% loss → halt trading. Max 3 consecutive losses → 4h cooldown.

**Per Week**: -7% → halt until Monday.

**Per Month**: -12% → bot OFF 14 days.

**Systemic**: VIX > 40 → flatten leveraged. SPY -5% → panic. API errors → failsafe close.

Dynamic risk calculation:
```python
base = regime_risk[regime]  # 0.5%-2%
if last_5_win_rate < 40%: base *= 0.5
if last_5_win_rate > 70%: base *= 1.2
vol_ratio = current_atr / avg_atr_252d
base /= vol_ratio
return min(base, 0.02)
```

## 12. Daily Schedule (CET)

- 07:00 Morning briefing (macro, calendar, regime recalc)
- 15:30-16:00 US OPEN BLACKOUT
- 16:00-21:30 Active trading window
- 21:30-22:00 Pre-close blackout
- 22:00 Daily reconciliation
- 22:00-07:00 Sleep mode (monitor only)

## 13. LLM Integration (Opt-In, Phase 2)

Config: `llm_enabled: false` (default)

3 functions:
1. **Daily Macro Brief** (07:00): 30-50 headlines → regime_bias, confidence, blackout_recommended
2. **Crash Narrative Detector** (every 30 min): scans for panic language → triggers panic mode if >70% confidence
3. **Event Filter** (weekly): earnings + macro events → pre-blackout 24h before

## 14. Mode Switch

```yaml
mode: "mini"   # no options, instruments: SPY/UPRO/SH/SPXS
mode: "full"   # with SPY options, VXX, capital >= 15k required
```

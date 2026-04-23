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

## 6. LONG Entry Signals (3-Level Check)

**Level 1 — SETUP (ALL required)**
- Regime score >= 8
- SPY > 200 SMA
- No major event in 24h (FOMC, CPI, NFP)
- ATR(14) < 3% (not chaotic)
- No opposing SHORT position open

**Level 2 — TRIGGER (min 3 of 5)**
1. RSI(14) 1H < 40, crossing up
2. Price touches 50 EMA and bounces
3. MACD histogram rising (from negative)
4. Stochastic(14,3,3) < 30, bullish cross
5. Bollinger Band lower touch + green candle

**Level 3 — CONFIRMATION (min 2 of 4)**
1. Volume > 1.3x 20-bar avg
2. Price above VWAP (5min)
3. A/D line rising
4. TICK index > +500 NYSE

**Asymmetric R:R**: Stop -1.5%, T1 +2%, T2 +4.5%, T3 +8%. Min R:R 1:2.0.

## 7. SHORT Entry Signals

**SETUP**: Regime score <= 5, VIX > 20 rising, SPY < 50 SMA, no panic rebound.

**TRIGGER (min 3 of 5)**: RSI > 68 crossing down, price rejected from 50 EMA, MACD bearish cross, Stoch > 75 bearish cross, BB upper rejection.

**CONFIRMATION (min 2 of 4)**: Red volume > green (3-bar), VIX rising >5% intraday, Put/Call > 1.0, A/D falling.

Execute SHORT via SPXS or SH — NOT naked short. Size 1/3 of normal long. R:R min 1:1.5.

## 8. Scale-in Logic (Anti-Martingale)

Max 6 positions, each 16.67% of allocated capital.
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

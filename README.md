# 🤖 SP500 Smart Scalper Bot

> **BOT DO ALGORYTMICZNEGO TRADINGU S&P 500 — TYLKO PAPER TRADING DOMYŚLNIE**

---

## ⚠️ OSTRZEŻENIE — PRZECZYTAJ PRZED UŻYCIEM

**TEN KOD TO OPROGRAMOWANIE EDUKACYJNE — NIE JEST PORADĄ FINANSOWĄ.**

- 📉 **Możesz stracić cały zainwestowany kapitał**
- 📊 Wyniki historyczne NIE gwarantują przyszłych zysków
- 🧪 Wymagane minimum 6 miesięcy paper tradingu przed przejściem na live
- 💼 Autor i asystent AI nie ponoszą odpowiedzialności za straty finansowe
- 🇨🇭 Użytkownicy w Szwajcarii: sprawdź regulacje FINMA przed live tradingiem
- 📋 Skonsultuj się z licencjonowanym doradcą finansowym

---

## 📋 Co robi ten bot?

SP500 Smart Scalper Bot to algorytmiczny bot tradingowy skupiony wyłącznie na S&P 500 (SPY, UPRO 3x, SH -1x, SPXS -3x). Używa 11-wskaźnikowego detektora reżimu rynkowego, systemu wejść 3-poziomowych, anti-martingale skalowania pozycji (do 6 jednocześnie), wielopoziomowego hedgingu oraz zabezpieczeń ryzyka. Docelowo: CAGR 25-35%, max drawdown -12%, Sharpe 1.5+. Domyślnie działa w trybie paper trading (symulacja) na IBKR.

*SP500 Smart Scalper Bot is an algorithmic trading bot focused exclusively on S&P 500 instruments. It uses an 11-indicator market regime detector, 3-level entry signals, anti-martingale position scaling (up to 6 simultaneous positions), multi-level hedging, and robust risk guards. Target: 25-35% CAGR, max -12% drawdown, Sharpe 1.5+. Defaults to paper trading mode.*

---

## 🚀 Szybki start (Paper Trading w 5 minut)

### 1. Wymagania
- Python 3.10+
- Interactive Brokers TWS/Gateway (paper account)
- Git

### 2. Instalacja

```bash
git clone https://github.com/ico2018konrad-design/SP500_smart.git
cd SP500_smart
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Konfiguracja

```bash
cp .env.example .env
# Edytuj .env — dodaj klucz FRED_API_KEY (darmowy: https://fred.stlouisfed.org/docs/api/api_key.html)
```

### 4. Uruchomienie backtrestu (bez IBKR, tylko yfinance)

```bash
python src/backtest/honest_backtest.py
```

### 5. Stres testy

```bash
python src/backtest/stress_test_2008.py
python src/backtest/stress_test_2020.py
python src/backtest/stress_test_2022.py
```

### 6. Dashboard Streamlit

```bash
streamlit run src/monitoring/dashboard.py
```

### 7. Uruchomienie bota (paper trading)

```bash
# Upewnij się że TWS/Gateway jest uruchomiony na porcie 7497
python src/main.py
```

### 8. Testy jednostkowe

```bash
pytest tests/ -v
```

---

## ⚙️ Konfiguracja

Główne pliki konfiguracyjne:

| Plik | Opis |
|------|------|
| `config/strategy_config.yaml` | Tryb (mini/full), progi reżimów, alokacja |
| `config/risk_limits.yaml` | Circuit breakery, limity dzienne/tygodniowe |
| `config/instruments.yaml` | Definicje instrumentów |
| `.env` | Klucze API, ustawienia IBKR |

### Przełączniki trybu

```yaml
# config/strategy_config.yaml
mode: "mini"          # "mini" (no options) or "full" (with options)
trading_mode: "paper"  # "paper" or "live" — LIVE WYŁĄCZONE DOMYŚLNIE
llm_enabled: false     # true = włącz AI regime detection (OpenAI/Anthropic)
starting_capital: 5000 # CHF
```

**⚠️ LIVE TRADING:** Zmień `trading_mode: "live"` tylko po 6+ miesiącach paper tradingu!

---

## 📊 Strategia — Skrót

### 5 warstw architektury
1. **Regime Detector** — 11 wskaźników → score 0-11 → 5 reżimów
2. **Signal Engine** — LONG/SHORT z 3-poziomową weryfikacją
3. **Position Manager** — max 6 pozycji, anti-martingale
4. **Hedge Module** — 3 poziomy (baseline/reactive/panic)
5. **Risk Guard** — circuit breakery, kill switch

### Reżimy rynkowe
| Score | Reżim | Max Leverage |
|-------|-------|-------------|
| 10-11 | STRONG BULL 🚀 | 2.5x |
| 8-9 | BULL 📈 | 2.0x |
| 6-7 | CHOP ↔️ | 1.0x |
| 4-5 | CAUTION ⚠️ | 0.5x |
| 0-3 | BEAR 📉 | 0x (short bias) |

### Zabezpieczenia
- Daily -3% → stop tradingu do następnego dnia
- Weekly -7% → stop do poniedziałku
- Monthly -12% → bot wyłączony na 14 dni
- VIX > 40 → flatten wszystkie pozycje lewarowane
- SPY -2% w godzinę → FLAT ALL (flash crash)
- Kill switch (manualny + automatyczny)

---

## 📅 Plan wdrożenia (5 faz)

Zobacz [PLAN.md](PLAN.md) po szczegóły.

| Faza | Opis | Czas | Warunek przejścia |
|------|------|------|-------------------|
| 1 | Setup + konfiguracja | 1 tydzień | Bot uruchomiony, testy przechodzą |
| 2 | Paper Mini (bez opcji) | 3 miesiące | Sharpe > 1.0, DD < 10% |
| 3 | Paper Full (z opcjami) | 3 miesiące | Sharpe > 1.3, DD < 8% |
| 4 | Live Mini (5k CHF) | 6 miesięcy | Sharpe > 1.2, tracking paper |
| 5 | Scale Up (15k+ CHF) | ciągłe | 15k+ CHF, Sharpe > 1.4 |

---

## 📁 Struktura projektu

```
SP500_smart/
├── config/          # Konfiguracja strategii i ryzyka
├── src/
│   ├── data/        # Loadery danych (Yahoo, IBKR, FRED)
│   ├── regime/      # Detektor reżimu (11 wskaźników)
│   ├── signals/     # Silnik sygnałów (LONG/SHORT)
│   ├── positions/   # Zarządzanie pozycjami
│   ├── hedge/       # System hedgingu
│   ├── risk/        # Zarządzanie ryzykiem
│   ├── execution/   # Egzekucja (paper/live)
│   ├── backtest/    # Backtesting + walk-forward
│   ├── monitoring/  # Dashboard, alerty, raporty
│   ├── schedule/    # Harmonogram dzienny
│   ├── llm/         # Opcjonalne AI (opt-in)
│   └── main.py      # Punkt startowy
├── tests/           # Testy jednostkowe
├── notebooks/       # Jupyter notebooki
├── docs/            # Dokumentacja
└── scripts/         # Skrypty pomocnicze
```

---

## 🔑 Wymagane klucze API

| Klucz | Wymagany | Gdzie dostać |
|-------|----------|-------------|
| `FRED_API_KEY` | ✅ Tak (backtest) | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) — bezpłatny |
| `IBKR_*` | ✅ Dla paper/live | IBKR TWS/Gateway |
| `OPENAI_API_KEY` | ❌ Opcjonalny | Tylko gdy `llm_enabled: true` |
| `ANTHROPIC_API_KEY` | ❌ Opcjonalny | Tylko gdy `llm_enabled: true` |
| `TELEGRAM_*` | ❌ Opcjonalny | Alerty telegram |

---

## 📚 Dokumentacja

- [STRATEGY.md](STRATEGY.md) — Pełna specyfikacja strategii (EN)
- [docs/STRATEGY_PL.md](docs/STRATEGY_PL.md) — Strategia po polsku
- [PLAN.md](PLAN.md) — Plan wdrożenia 5 faz
- [docs/SETUP_IBKR.md](docs/SETUP_IBKR.md) — Konfiguracja IBKR
- [docs/FAQ.md](docs/FAQ.md) — Często zadawane pytania

---

## 🛡️ Disclaimer (EN)

This software is provided for educational purposes only. It does not constitute financial advice. Trading involves substantial risk of loss. Past performance does not guarantee future results. The authors and contributors are not liable for any financial losses incurred through use of this software. Always consult a licensed financial advisor before trading with real money. Paper trade for at least 6 months before considering live trading.

---

*Inspired by a friend's system that achieved 196% in 2025 — with key improvements: Valuation Guard, Walk-Forward Validation, ATR-based sizing, and optional LLM regime detection.*

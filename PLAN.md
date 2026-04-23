# 📅 SP500 Smart Scalper Bot — Plan Wdrożenia (5 Faz)

## Faza 1: Setup & Konfiguracja (1 tydzień)

### Cel
Uruchomienie bota w środowisku paper trading, weryfikacja techniczna.

### Zadania
- [ ] Instalacja Python 3.10+, dependencies
- [ ] Konfiguracja IBKR TWS/Gateway (paper account)
- [ ] Ustawienie .env (FRED_API_KEY minimum)
- [ ] Uruchomienie backtrestu 2005-2025
- [ ] Uruchomienie stress testów (2008/2020/2022)
- [ ] Weryfikacja: `pytest tests/` — wszystkie testy przechodzą
- [ ] Uruchomienie dashboardu Streamlit
- [ ] Przegląd walk-forward validation (16 okien)

### Kryteria wyjścia
- ✅ Bot uruchamia się bez błędów
- ✅ Backtest wyświetla wyniki
- ✅ Dashboard działa
- ✅ Testy: 100% pass

---

## Faza 2: Paper Mini Mode (3 miesiące)

### Cel
Walidacja strategii w warunkach rynkowych bez opcji, małym kapitałem.

### Konfiguracja
```yaml
mode: "mini"
trading_mode: "paper"
starting_capital: 5000
```

### Zadania
- [ ] Paper trading przez min. 3 miesiące
- [ ] Minimum 50 transakcji
- [ ] Codzienny przegląd logu trades
- [ ] Tygodniowe review equity curve
- [ ] Porównanie z benchmark (SPY buy&hold)

### Kryteria wyjścia (ALL wymagane)
- ✅ Sharpe Ratio > 1.0 (annualized)
- ✅ Max Drawdown < 10%
- ✅ Win Rate > 45%
- ✅ Profit Factor > 1.3
- ✅ Min 50 zamkniętych transakcji
- ✅ Brak błędów systemowych przez ostatnie 30 dni
- ✅ Reżim i sygnały zachowują się zgodnie ze strategią

---

## Faza 3: Paper Full Mode (3 miesiące)

### Cel
Włączenie trybu pełnego z opcjami (gdy kapitał > 15k paper), LLM (opcjonalnie).

### Konfiguracja
```yaml
mode: "full"
trading_mode: "paper"
llm_enabled: true  # opcjonalnie
```

### Zadania
- [ ] Upgrade do trybu "full" gdy paper equity > 15k
- [ ] Weryfikacja opcji SPY put spreads (paper)
- [ ] Włączenie LLM module (opcjonalnie) — A/B test
- [ ] Porównanie Sharpe z/bez LLM
- [ ] Weryfikacja hedge'u opcyjnego

### Kryteria wyjścia
- ✅ Sharpe > 1.3
- ✅ Max Drawdown < 8%
- ✅ Hedge skutecznie redukuje DD w volatilne dni
- ✅ LLM nie powoduje false-positive sygnałów > 20%

---

## Faza 4: Live Mini (6 miesięcy)

### Cel
Pierwsze środki na żywo, małe pozycje, maksymalna ostrożność.

### ⚠️ UWAGA: Tylko po przejściu Fazy 1-3!

### Konfiguracja
```yaml
mode: "mini"
trading_mode: "live"
starting_capital: 5000  # CHF real
```

### Warunki konieczne
- ✅ 6 miesięcy paper tradingu zakończone
- ✅ Faza 2 + 3 kryteria spełnione
- ✅ Pełne zrozumienie wszystkich mechanizmów
- ✅ Konto IBKR Switzerland aktywne
- ✅ Kill switch przetestowany
- ✅ Telegram alerty działają

### Zarządzanie ryzykiem w Fazie 4
- Max pozycja: 10% kapitału (500 CHF)
- Max dzienny loss: 2% (100 CHF)
- Kill switch przy -5% (250 CHF) — manualny przegląd

### Kryteria wyjścia
- ✅ Sharpe > 1.2 (6 miesięcy live)
- ✅ Max Drawdown < 10%
- ✅ Kapitał wzrósł do 7k+ CHF
- ✅ Live performance śledzi paper ± 20%

---

## Faza 5: Scale Up (ciągłe)

### Cel
Systematyczne zwiększanie kapitału, przejście na tryb full z opcjami.

### Kamienie milowe
| Kapitał | Akcja |
|---------|-------|
| 7k CHF | Zwiększ pozycje o 20% |
| 10k CHF | Rozważ tryb Full bez opcji |
| 15k CHF | Włącz opcje SPY put spreads |
| 25k CHF | Pełna strategia, all features enabled |
| 50k+ CHF | Rozważ dywersyfikację do innych instrumentów |

### Zasady skalowania
- **NIE** zwiększaj kapitału po stratnych tygodniach
- Skaluj tylko gdy: Sharpe > 1.3, DD < 8% (rolling 3 miesiące)
- Nigdy nie przekraczaj 25% kapitału w jednej pozycji
- Utrzymuj min. 20% cash buffer zawsze

---

## 📊 Metryki do monitorowania (wszystkie fazy)

| Metryka | Cel | Alert |
|---------|-----|-------|
| CAGR | 25-35% | < 10% → review |
| Sharpe Ratio | > 1.5 | < 1.0 → pause |
| Max Drawdown | < 12% | > 8% → reduce size |
| Win Rate | > 50% | < 40% → review signals |
| Profit Factor | > 1.5 | < 1.2 → pause |
| Avg R:R | > 2.0 | < 1.5 → review exits |

---

*Ten plan jest inspirowany strategią znajomego, który osiągnął 196% w 2025 roku, z kluczowymi ulepszeniami: Valuation Guard, Walk-Forward Validation, ATR sizing, opcjonalny LLM.*

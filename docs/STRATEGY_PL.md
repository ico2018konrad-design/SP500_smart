# SP500 Smart Scalper Bot — Strategia Handlowa (PL)

## 1. Filozofia i Cele

Bot skupia się wyłącznie na S&P 500. Filozofia:
- **Nastawienie długie**: rynek rośnie 75% czasu, short tylko w potwierdzonym niedźwiedziu
- **Hedge ZAWSZE aktywny**: ubezpieczenie baseline w każdych warunkach
- **Dźwignia tylko gdy potwierdzony BULL**: wiele wskaźników musi potwierdzić
- **Przeżycie > maksymalny zysk**: ochrona kapitału to priorytet

**Cel:** CAGR 25-35%, max DD -12%, Sharpe 1.5+

## 2. Reżimy Rynkowe

11-wskaźnikowy detektor reżimu:

| Wynik | Reżim | Max Dźwignia |
|-------|-------|-------------|
| 10-11 | STRONG BULL 🚀 | 2.5x |
| 8-9   | BULL 📈 | 2.0x |
| 6-7   | CHOP ↔️ | 1.0x |
| 4-5   | CAUTION ⚠️ | 0.5x |
| 0-3   | BEAR 📉 | 0x |

## 3. Walidacja Wyceny (Valuation Guard)

KLUCZOWA INNOWACJA — korekcja na szczycie rynku:

```python
penalty = 1.0
if spy > sma200 * 1.15:  penalty *= 0.75  # przegrzanie
if CAPE > 35:            penalty *= 0.75  # wysokie wyceny
if VIX < 13:             penalty *= 0.80  # samozadowolenie
if mies_bez_korekty > 18: penalty *= 0.70 # zaległa korekta
dźwignia_final = dźwignia_bazowa × penalty
```

## 4. Anti-Martingale (Skalowanie Pozycji)

**Nigdy nie uśredniaj straty!** Dodaj pozycję TYLKO gdy poprzednia jest na +1%:

```
Poz #1 otwarta @ 400$
→ +1% (404$): Otwórz Poz #2, przesuń stop Poz#1 do break-even
→ +1% Poz#2: Otwórz Poz #3, podnoś stopy
→ ... do max 6 pozycji
```

## 5. Zarządzanie Ryzykiem (5 poziomów)

- **Dzienny**: -3% → stop do następnego dnia
- **Tygodniowy**: -7% → stop do poniedziałku
- **Miesięczny**: -12% → bot wyłączony na 14 dni
- **VIX > 40**: flatuj wszystkie lewarowane pozycje
- **SPY -2% w godzinę**: FLATUJ WSZYSTKO (flash crash)

## 6. Ostrzeżenie

⚠️ Ten bot jest oprogramowaniem edukacyjnym. Możesz stracić cały kapitał.
Paper trading MINIMUM 6 miesięcy przed live tradingiem.

# Konfiguracja Interactive Brokers (IBKR)

## 1. Konto Paper Trading

1. Wejdź na: https://www.interactivebrokers.com
2. Utwórz konto (Paper Trading dostępne bez depozytu)
3. Zaloguj się do Client Portal
4. Aktywuj Paper Trading Account

## 2. TWS / IB Gateway

Pobierz Trader Workstation (TWS) lub IB Gateway:
- TWS: https://www.interactivebrokers.com/en/trading/tws.php
- IB Gateway (lżejsza wersja): https://www.interactivebrokers.com/en/trading/ibgateway-latest.php

## 3. Konfiguracja API

W TWS: File → Global Configuration → API → Settings:
- ✅ Enable ActiveX and Socket Clients
- Socket Port: **7497** (paper) lub **7496** (live)
- ✅ Allow connections from localhost only

## 4. .env konfiguracja

```bash
IBKR_HOST=127.0.0.1
IBKR_PORT=7497        # paper trading
IBKR_CLIENT_ID=1
```

## 5. Test połączenia

```python
from src.data.ibkr_loader import IBKRLoader
loader = IBKRLoader()
loader.connect()
print(loader.get_account_summary())
loader.disconnect()
```

## 6. Ważne porty

| Tryb | Port TWS | Port Gateway |
|------|----------|--------------|
| Paper | 7497 | 4002 |
| Live | 7496 | 4001 |

⚠️ Używaj TYLKO portu 7497 (paper) dopóki nie zakończysz 6 miesięcy testów!

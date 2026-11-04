# market-recap-data

End-of-day US market data for a personal daily recap, updated by GitHub Actions.

- `data/latest.json`: the most recent completed session
- `data/history/`: one file per session
- `scripts/fetch_market_data.py`: the fetch script

Prices come from Yahoo Finance through the unofficial `yfinance` package, Treasury yields from the US Treasury, and index constituents from Wikipedia. For personal information only.

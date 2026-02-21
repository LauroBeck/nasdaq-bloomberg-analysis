import yfinance as yf
import pandas as pd

# --------------------------------
# Tickers (Yahoo Finance symbols)
# --------------------------------
tickers = {
    "Chevron": "CVX",
    "ConocoPhillips": "COP",
    "ExxonMobil": "XOM",
    "Shell": "SHEL",
    "BP": "BP",
    "Halliburton": "HAL",
    "Baker Hughes": "BKR",
    "Maersk": "AMKBY",        # US ADR
    "Hapag-Lloyd": "HPGLY"    # US ADR
}

start_date = "2024-01-01"
end_date = "2025-12-31"

# --------------------------------
# Download Data
# --------------------------------
data = yf.download(
    list(tickers.values()),
    start=start_date,
    end=end_date,
    group_by="ticker",
    auto_adjust=True
)

# --------------------------------
# Build Clean Close-Price Table
# --------------------------------
close_prices = pd.DataFrame()

for company, ticker in tickers.items():
    close_prices[company] = data[ticker]["Close"]

print("\nLatest Prices\n")
print(close_prices.tail())

# Optional: Save to CSV
close_prices.to_csv("market_prices.csv")

print("\nSaved to market_prices.csv")

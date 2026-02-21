import yfinance as yf
import pandas as pd

# Data mapping from the Bloomberg image (Feb 11, 2026)
market_data = {
    "Index": ["S&P 500", "NASDAQ", "KOSPI", "NIFTY 50", "ASX 200"],
    "Image Value": [6943.22, 23067.27, 5354.49, 25953.85, 9014.78],
    "Ticker": ["^GSPC", "^IXIC", "^KS11", "^NSEI", "^AXJO"]
}

# 1. Create a reference DataFrame
df_reference = pd.DataFrame(market_data)
print("--- Reference Data from Image ---")
print(df_reference)
print("\n")

# 2. Fetch live data for these tickers
print("--- Fetching Live Data ---")
tickers = market_data["Ticker"]
data = yf.download(tickers, period="1d", interval="1m")

# Check for 'Close' or 'Adj Close' based on yfinance version
price_col = 'Close' if 'Close' in data.columns else 'Adj Close'

# Display latest available prices for each ticker
for ticker in tickers:
    try:
        latest_price = data[price_col][ticker].dropna().iloc[-1]
        print(f"{ticker}: {latest_price:,.2f}")
    except Exception as e:
        print(f"Could not retrieve live data for {ticker}")


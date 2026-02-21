import yfinance as yf
import pandas as pd

# Tickers for the indices shown in the image
# S&P 500 (^GSPC), NASDAQ Composite (^IXIC), KOSPI (^KS11), NIFTY 50 (^NSEI), S&P/ASX 200 (^AXJO)
tickers = ["^GSPC", "^IXIC", "^KS11", "^NSEI", "^AXJO"]

# Download data for February 11, 2026 (data from the image)
# Note: yfinance requires 'end' to be one day after the desired date
data = yf.download(tickers, start="2026-02-11", end="2026-02-12")

# Check if 'Adj Close' exists (older versions) or use 'Close' (newer versions with auto_adjust)
if 'Adj Close' in data.columns:
    prices = data['Adj Close']
else:
    print("Using 'Close' column (Standard in newer yfinance versions)")
    prices = data['Close']

# Display the values matching the image
print(prices)

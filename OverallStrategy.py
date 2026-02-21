import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

tickers = {
    "SK hynix": "000660.KQ",
    "TSMC": "TSM",
    "Dell": "DELL",
    "IBM": "IBM"
}

for name, symbol in tickers.items():
    print(f"\n=== {name} ({symbol}) ===")
    
    ticker = yf.Ticker(symbol)
    
    # Historical prices (5y)
    hist = ticker.history(period="5y")
    
    # Annualized dividend calculation
    # Sum last 12 months of dividends
    if not hist["Dividends"].empty:
        annual_div = hist["Dividends"].rolling(252).sum().iloc[-1]  # Approx 252 trading days
    else:
        annual_div = 0
    
    latest_price = hist["Close"].iloc[-1]
    dividend_yield = (annual_div / latest_price) * 100 if annual_div > 0 else 0
    print(f"Latest Price: ${latest_price:.2f}, Annual Dividend: ${annual_div:.2f}, Yield: {dividend_yield:.2f}%")
    
    # SMA calculation
    hist["SMA20"] = hist["Close"].rolling(20).mean()
    hist["SMA50"] = hist["Close"].rolling(50).mean()
    
    # Signals based on dividend yield vs historical mean
    if annual_div > 0:
        avg_yield = (hist["Dividends"].rolling(252).sum() / hist["Close"] * 100).mean()
        hist["Signal"] = 0
        hist.loc[(hist["Dividends"].rolling(252).sum() / hist["Close"] * 100) > avg_yield, "Signal"] = 1
        hist.loc[(hist["Dividends"].rolling(252).sum() / hist["Close"] * 100) < avg_yield, "Signal"] = -1
    else:
        hist["Signal"] = 0
    
    # Save for BI dashboard
    hist.to_csv(f"{symbol}_dividend_bi.csv")

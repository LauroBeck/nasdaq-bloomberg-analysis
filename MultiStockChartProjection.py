import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Tickers for analysis
tickers = {
    "JPM": "JPM",
    "SK hynix": "000660.KQ",
    "TSMC": "TSM",
    "Dell": "DELL",
    "IBM": "IBM"
}

shares_held = 1000  # Example: 1000 shares for BI projection

for name, symbol in tickers.items():
    print(f"\n=== {name} ({symbol}) ===")
    ticker = yf.Ticker(symbol)
    
    # Historical data
    hist = ticker.history(period="5y")
    if hist.empty:
        print(f"No data for {name}. Skipping.")
        continue
    
    # Annualized dividend (last 252 trading days)
    hist["AnnualDiv"] = hist["Dividends"].rolling(252).sum()
    hist["DividendYield"] = (hist["AnnualDiv"] / hist["Close"]) * 100
    
    # SMA calculation
    hist["SMA20"] = hist["Close"].rolling(20).mean()
    hist["SMA50"] = hist["Close"].rolling(50).mean()
    
    # Buy/Sell signals based on dividend yield vs historical average
    if hist["DividendYield"].notnull().any():
        avg_yield = hist["DividendYield"].mean()
        hist["Signal"] = 0
        hist.loc[hist["DividendYield"] > avg_yield, "Signal"] = 1   # Buy
        hist.loc[hist["DividendYield"] < avg_yield, "Signal"] = -1  # Sell
    else:
        hist["Signal"] = 0
        avg_yield = 0
    
    # Dividend Yield Chart
    plt.figure(figsize=(12,5))
    plt.plot(hist.index, hist["DividendYield"], label="Dividend Yield (%)")
    plt.axhline(avg_yield, color="red", linestyle="--", label="Avg Yield")
    plt.scatter(hist.index[hist["Signal"]==1], hist["DividendYield"][hist["Signal"]==1],
                marker="^", color="green", label="Buy Signal", s=100)
    plt.scatter(hist.index[hist["Signal"]==-1], hist["DividendYield"][hist["Signal"]==-1],
                marker="v", color="red", label="Sell Signal", s=100)
    plt.title(f"{name} Dividend Yield Projection")
    plt.xlabel("Date")
    plt.ylabel("Dividend Yield (%)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Price Chart with SMA
    plt.figure(figsize=(12,5))
    plt.plot(hist.index, hist["Close"], label="Close Price")
    plt.plot(hist.index, hist["SMA20"], label="SMA20")
    plt.plot(hist.index, hist["SMA50"], label="SMA50")
    plt.title(f"{name} Price & SMA Projection")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # BI Projection: Annual dividend income for shares_held
    latest_price = hist["Close"].iloc[-1]
    last_annual_div = hist["AnnualDiv"].iloc[-1]
    projected_income = last_annual_div * shares_held
    print(f"Latest Price: ${latest_price:.2f}")
    print(f"Projected Annual Dividend for {shares_held} shares: ${projected_income:.2f}")
    
    # Save CSV for BI dashboards
    hist.to_csv(f"{symbol}_dividend_projection.csv")
    print(f"Saved data to {symbol}_dividend_projection.csv")

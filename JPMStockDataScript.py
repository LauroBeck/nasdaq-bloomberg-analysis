# JPM Stock Data Script
# Requires: yfinance, pandas, matplotlib

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def fetch_jpm_data():
    # Initialize ticker
    ticker = yf.Ticker("JPM")

    # 1️⃣ Latest price and info
    info = ticker.info
    latest_price = info.get("regularMarketPrice", None)
    print(f"Latest JPM price: ${latest_price:.2f}")
    print("52-week range:", info.get("fiftyTwoWeekLow"), "-", info.get("fiftyTwoWeekHigh"))
    print("Dividend yield:", info.get("dividendYield"))

    # 2️⃣ Historical data (last 2 years)
    hist = ticker.history(period="2y")
    print("\nHistorical data sample:")
    print(hist.head())

    # 3️⃣ Calculate moving averages
    hist["SMA20"] = hist["Close"].rolling(20).mean()
    hist["SMA50"] = hist["Close"].rolling(50).mean()

    # 4️⃣ Plot closing price and moving averages
    plt.figure(figsize=(12, 6))
    plt.plot(hist["Close"], label="Close")
    plt.plot(hist["SMA20"], label="20-day SMA")
    plt.plot(hist["SMA50"], label="50-day SMA")
    plt.title("JPM Close Price with 20 & 50-day SMA")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 5️⃣ Save historical data to CSV
    hist.to_csv("jpm_historical.csv")
    print("\nHistorical data saved to 'jpm_historical.csv'")

if __name__ == "__main__":
    fetch_jpm_data()

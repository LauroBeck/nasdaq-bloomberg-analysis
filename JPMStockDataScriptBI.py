# JPM Dividend Yield Strategy & BI Projection
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def jpm_dividend_strategy():
    ticker = yf.Ticker("JPM")
    
    # Fetch historical data
    hist = ticker.history(period="5y")  # 5 years to see trend

    # 1️⃣ Calculate Dividend Yield over time
    # Annual Dividend per share (using dividends column)
    hist["AnnualDiv"] = hist["Dividends"].rolling(252).sum()  # 252 trading days ~ 1 year
    hist["DividendYield"] = hist["AnnualDiv"] / hist["Close"] * 100  # in %

    # Historical average yield
    avg_yield = hist["DividendYield"].mean()
    print(f"Historical Average Dividend Yield: {avg_yield:.2f}%")
    
    # Buy/Sell signals based on DY
    hist["Signal"] = 0
    hist.loc[hist["DividendYield"] > avg_yield, "Signal"] = 1  # Buy signal
    hist.loc[hist["DividendYield"] < avg_yield, "Signal"] = -1 # Sell signal

    # Plot Dividend Yield & Signals
    plt.figure(figsize=(12,6))
    plt.plot(hist.index, hist["DividendYield"], label="Dividend Yield (%)")
    plt.axhline(avg_yield, color="red", linestyle="--", label="Avg DY")
    plt.scatter(hist.index[hist["Signal"]==1], hist["DividendYield"][hist["Signal"]==1],
                marker="^", color="green", label="Buy Signal", s=100)
    plt.scatter(hist.index[hist["Signal"]==-1], hist["DividendYield"][hist["Signal"]==-1],
                marker="v", color="red", label="Sell Signal", s=100)
    plt.title("JPM Dividend Yield Strategy")
    plt.xlabel("Date")
    plt.ylabel("Dividend Yield (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2️⃣ Business Intelligence Projection: Estimated Annual Dividend Income
    latest_price = ticker.info["regularMarketPrice"]
    latest_dividend = hist["Dividends"][-252:].sum()  # last year dividend
    projected_annual_income = latest_dividend / latest_price * 100
    print(f"Latest Price: ${latest_price:.2f}")
    print(f"Last Year Dividend: ${latest_dividend:.2f}")
    print(f"Projected Annual Dividend Yield: {projected_annual_income:.2f}%")

    # Save to CSV
    hist.to_csv("jpm_dividend_strategy.csv")
    print("Dividend strategy data saved to 'jpm_dividend_strategy.csv'")

if __name__ == "__main__":
    jpm_dividend_strategy()

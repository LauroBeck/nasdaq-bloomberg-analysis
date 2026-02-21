import pandas as pd
import matplotlib.pyplot as plt

# Load your historical data
df = pd.read_csv("jpm_feb2026.csv", parse_dates=["Date"])
df.sort_values("Date", inplace=True)

# Example: JPM annual dividend (approx from last 12 months)
annual_dividend = 6.0  # USD per share
df["DividendYield"] = annual_dividend / df["Close"] * 100  # in %

# Historical average dividend yield
avg_yield = df["DividendYield"].mean()
print(f"Average Dividend Yield (Feb 2026 sample): {avg_yield:.2f}%")

# Generate buy/sell signals
df["Signal"] = 0
df.loc[df["DividendYield"] > avg_yield, "Signal"] = 1  # Buy
df.loc[df["DividendYield"] < avg_yield, "Signal"] = -1 # Sell

# Calculate moving averages (SMA20, SMA50)
df["SMA20"] = df["Close"].rolling(20).mean()
df["SMA50"] = df["Close"].rolling(50).mean()

# Plot Dividend Yield + Signals
plt.figure(figsize=(12,6))
plt.plot(df["Date"], df["DividendYield"], label="Dividend Yield (%)")
plt.axhline(avg_yield, color="red", linestyle="--", label="Avg Yield")
plt.scatter(df["Date"][df["Signal"]==1], df["DividendYield"][df["Signal"]==1],
            marker="^", color="green", label="Buy Signal", s=100)
plt.scatter(df["Date"][df["Signal"]==-1], df["DividendYield"][df["Signal"]==-1],
            marker="v", color="red", label="Sell Signal", s=100)
plt.title("JPM Dividend Yield Strategy")
plt.xlabel("Date")
plt.ylabel("Dividend Yield (%)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Close Price with SMAs
plt.figure(figsize=(12,6))
plt.plot(df["Date"], df["Close"], label="Close Price")
plt.plot(df["Date"], df["SMA20"], label="SMA20")
plt.plot(df["Date"], df["SMA50"], label="SMA50")
plt.title("JPM Price with 20/50-Day SMA")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.show()

# BI Projection: Annual Dividend Income per 1000 shares
latest_price = df["Close"].iloc[-1]
projected_income = (1000 * annual_dividend)
print(f"Latest Price: ${latest_price:.2f}")
print(f"Projected Annual Dividend for 1000 shares: ${projected_income:.2f}")

# Save to CSV for BI dashboard
df.to_csv("jpm_dividend_bi.csv", index=False)
print("Saved dividend strategy data to 'jpm_dividend_bi.csv'")

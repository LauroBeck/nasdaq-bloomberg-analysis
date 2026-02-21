# JPM Dividend Yield Strategy & BI Projection
import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ Historical data directly in script
data = {
    "Date": ["2026-02-13","2026-02-12","2026-02-11","2026-02-10","2026-02-09",
             "2026-02-06","2026-02-05","2026-02-04","2026-02-03","2026-02-02"],
    "Close": [302.55,302.64,310.82,318.28,322.10,322.40,310.16,317.27,314.85,308.14],
    "Volume": [9114526,13443360,8703519,9902224,11477470,17797440,9387999,9848802,12687960,9839360],
    "Open": [298.52,312.275,323.24,322.55,321.34,314.71,315.00,314.405,309.73,304.46],
    "High": [304.29,313.615,325.28,326.125,326.40,324.245,316.01,319.305,316.25,309.30],
    "Low": [296.52,300.02,308.7301,315.1201,320.1149,314.71,305.54,314.405,309.095,301.37]
}

df = pd.DataFrame(data)
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)

# 2️⃣ Dividend Yield Calculation
annual_dividend = 6.0  # USD per share (approx last year)
df["DividendYield"] = annual_dividend / df["Close"] * 100  # %

# Historical average
avg_yield = df["DividendYield"].mean()
print(f"Average Dividend Yield: {avg_yield:.2f}%")

# 3️⃣ Buy/Sell Signals
df["Signal"] = 0
df.loc[df["DividendYield"] > avg_yield, "Signal"] = 1   # Buy
df.loc[df["DividendYield"] < avg_yield, "Signal"] = -1  # Sell

# 4️⃣ Moving Averages
df["SMA20"] = df["Close"].rolling(20).mean()
df["SMA50"] = df["Close"].rolling(50).mean()

# 5️⃣ Plot Dividend Yield + Signals
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

# 6️⃣ Plot Close Price with SMAs
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

# 7️⃣ Business Intelligence Projection
latest_price = df["Close"].iloc[-1]
projected_income = 1000 * annual_dividend  # for 1000 shares
print(f"Latest Price: ${latest_price:.2f}")
print(f"Projected Annual Dividend for 1000 shares: ${projected_income:.2f}")

# 8️⃣ Save to CSV
df.to_csv("jpm_dividend_bi.csv", index=False)
print("Saved dividend strategy data to 'jpm_dividend_bi.csv'")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURATION
# -----------------------------

TICKER = "IBM"
PERIOD = "2y"
INTERVAL = "1d"
TRADING_DAYS = 252

# -----------------------------
# DOWNLOAD DATA
# -----------------------------

print(f"Downloading data for {TICKER}...")

df = yf.download(TICKER, period=PERIOD, interval=INTERVAL)
df = df.dropna()

if df.empty:
    raise ValueError("No data retrieved. Check ticker or connection.")

# -----------------------------
# NORMALIZE PRICE SERIES (CRITICAL FIX)
# -----------------------------

close = df["Close"]

# yfinance / pandas compatibility guard
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

close = close.astype(float)

# -----------------------------
# BLOOMBERG-STYLE METRICS
# -----------------------------

# Returns
df["returns"] = close.pct_change()

# Momentum (12M)
df["momentum_12m"] = close / close.shift(TRADING_DAYS) - 1

# Rolling volatility
df["volatility"] = df["returns"].rolling(TRADING_DAYS).std() * np.sqrt(TRADING_DAYS)

# Rolling mean return
df["mean_return"] = df["returns"].rolling(TRADING_DAYS).mean() * TRADING_DAYS

# Drawdown (now guaranteed safe)
df["cum_max"] = close.cummax()
df["drawdown"] = close / df["cum_max"] - 1

# -----------------------------
# BI SNAPSHOT
# -----------------------------

latest = df.iloc[-1]

print("\nIBM BLOOMBERG-STYLE BI SNAPSHOT")
print("-----------------------------------")
print(f"Last Price:           {close.iloc[-1]:.2f} USD")
print(f"12M Momentum:         {latest['momentum_12m']:.2%}")
print(f"Annual Volatility:    {latest['volatility']:.2%}")
print(f"Rolling Mean Return:  {latest['mean_return']:.2%}")
print(f"Current Drawdown:     {latest['drawdown']:.2%}")

# -----------------------------
# FUNDAMENTAL / FACTOR OVERLAY
# -----------------------------

revenue_growth = 0.05
pe_ratio = 23.3
dividend_yield = 0.03
ai_bookings_growth = True

signals = {
    "revenue_growth": 1 if revenue_growth >= 0.05 else 0,
    "valuation": 1 if pe_ratio < 25 else 0,
    "income": 1 if dividend_yield > 0.025 else 0,
    "ai_momentum": 1 if ai_bookings_growth else 0
}

factor_score = sum(signals.values())

print("\nFACTOR MODEL")
print("-----------------------------------")
for k, v in signals.items():
    print(f"{k:20}: {v}")

print(f"\nComposite Factor Score: {factor_score}")

# -----------------------------
# RISK / RETURN ESTIMATES
# -----------------------------

expected_return = df["returns"].mean() * TRADING_DAYS
risk = df["returns"].std() * np.sqrt(TRADING_DAYS)

print("\nRISK / RETURN ESTIMATES")
print("-----------------------------------")
print(f"Expected Annual Return: {expected_return:.2%}")
print(f"Annualized Risk:        {risk:.2%}")

# -----------------------------
# VISUALIZATION
# -----------------------------

plt.figure()
plt.plot(close)
plt.title("IBM Price History")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.show()

plt.figure()
plt.plot(df["drawdown"])
plt.title("IBM Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.show()

plt.figure()
plt.plot(df["volatility"])
plt.title("IBM Rolling Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.show()

# -----------------------------
# OPTIMIZATION PREP
# -----------------------------

optimization_inputs = {
    "expected_return": expected_return,
    "risk": risk,
    "momentum": latest["momentum_12m"]
}

print("\nOPTIMIZATION INPUTS")
print("-----------------------------------")
for k, v in optimization_inputs.items():
    print(f"{k:20}: {float(v):.6f}")

print("\nDone.")


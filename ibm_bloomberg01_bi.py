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
# SCALAR SAFETY HELPER
# -----------------------------

def scalar(x):
    """Safely convert pandas objects or scalars to float."""
    if isinstance(x, pd.DataFrame):
        return float(x.iloc[0, 0])
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)

# -----------------------------
# DOWNLOAD DATA (Bloomberg HP proxy)
# -----------------------------

print(f"Downloading data for {TICKER}...")

df = yf.download(TICKER, period=PERIOD, interval=INTERVAL)
df = df.dropna()

if df.empty:
    raise ValueError("No data retrieved. Check ticker or connection.")

# -----------------------------
# NORMALIZE CLOSE PRICE
# -----------------------------

close = df["Close"]

if isinstance(close, pd.DataFrame):   # MultiIndex protection
    close = close.iloc[:, 0]

close = close.astype(float)

# -----------------------------
# BLOOMBERG-STYLE METRICS
# -----------------------------

df["returns"] = close.pct_change()

df["momentum_12m"] = close / close.shift(TRADING_DAYS) - 1

df["volatility"] = (
    df["returns"]
    .rolling(TRADING_DAYS)
    .std() * np.sqrt(TRADING_DAYS)
)

df["mean_return"] = (
    df["returns"]
    .rolling(TRADING_DAYS)
    .mean() * TRADING_DAYS
)

df["cum_max"] = close.cummax()
df["drawdown"] = close / df["cum_max"] - 1

latest = df.iloc[-1]

# -----------------------------
# BI SNAPSHOT
# -----------------------------

print("\nIBM BLOOMBERG-STYLE BI SNAPSHOT")
print("-----------------------------------")
print(f"Last Price:           {scalar(close.iloc[-1]):.2f} USD")
print(f"12M Momentum:         {scalar(latest['momentum_12m']):.2%}")
print(f"Annual Volatility:    {scalar(latest['volatility']):.2%}")
print(f"Rolling Mean Return:  {scalar(latest['mean_return']):.2%}")
print(f"Current Drawdown:     {scalar(latest['drawdown']):.2%}")

# -----------------------------
# FUNDAMENTAL / FACTOR OVERLAY
# (Manual proxies — Bloomberg normally supplies)
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

expected_return = scalar(df["returns"].mean() * TRADING_DAYS)
risk = scalar(df["returns"].std() * np.sqrt(TRADING_DAYS))

print("\nRISK / RETURN ESTIMATES")
print("-----------------------------------")
print(f"Expected Annual Return: {expected_return:.2%}")
print(f"Annualized Risk:        {risk:.2%}")

# -----------------------------
# SAVE CHARTS (No plt.show())
# -----------------------------

def save_chart(series, title, filename, ylabel):
    plt.figure()
    plt.plot(series)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

save_chart(close, "IBM Price History", "ibm_price.png", "Price (USD)")
save_chart(df["drawdown"], "IBM Drawdown", "ibm_drawdown.png", "Drawdown")
save_chart(df["volatility"], "IBM Rolling Volatility", "ibm_volatility.png", "Volatility")

print("\nCharts saved:")
print(" - ibm_price.png")
print(" - ibm_drawdown.png")
print(" - ibm_volatility.png")

# -----------------------------
# OPTIMIZATION INPUTS
# -----------------------------

optimization_inputs = {
    "expected_return": expected_return,
    "risk": risk,
    "momentum": scalar(latest["momentum_12m"])
}

print("\nOPTIMIZATION INPUTS")
print("-----------------------------------")
for k, v in optimization_inputs.items():
    print(f"{k:20}: {v:.6f}")

print("\nDone.")

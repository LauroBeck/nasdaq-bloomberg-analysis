import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURATION
# -----------------------------

TICKER = "JPMC34.SA"     # B3 BDR ticker
PERIOD = "2y"
INTERVAL = "1d"
TRADING_DAYS = 252

# -----------------------------
# SCALAR SAFETY HELPER
# -----------------------------

def scalar(x):
    if isinstance(x, pd.DataFrame):
        return float(x.iloc[0, 0])
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)

# -----------------------------
# DOWNLOAD DATA
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

if isinstance(close, pd.DataFrame):
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
# BI SNAPSHOT (Brazilian style)
# -----------------------------

print("\nJPMC34 BLOOMBERG-STYLE BI SNAPSHOT")
print("-----------------------------------")
print(f"Last Price:           {scalar(close.iloc[-1]):.2f} BRL")
print(f"12M Momentum:         {scalar(latest['momentum_12m']):.2%}")
print(f"Annual Volatility:    {scalar(latest['volatility']):.2%}")
print(f"Rolling Mean Return:  {scalar(latest['mean_return']):.2%}")
print(f"Current Drawdown:     {scalar(latest['drawdown']):.2%}")

# -----------------------------
# FUNDAMENTAL / NARRATIVE SIGNALS
# (BDRs often lack full fundamentals in yfinance)
# -----------------------------

earnings_miss = True
revenue_growth = 0.025      # +2.51% YoY proxy from your data
macro_financial_strength = True

signals = {
    "revenue_growth": 1 if revenue_growth > 0 else 0,
    "earnings_pressure": 0 if earnings_miss else 1,
    "financial_sector_strength": 1 if macro_financial_strength else 0,
}

factor_score = sum(signals.values())

print("\nFACTOR MODEL")
-----------------------------------
for k, v in signals.items():
    print(f"{k:30}: {v}")

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
# SAVE CHARTS (Headless safe)
# -----------------------------

def save_chart(series, title, filename, ylabel):
    plt.figure()
    plt.plot(series)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

save_chart(close, "JPMC34 Price History", "jpmc34_price.png", "Price (BRL)")
save_chart(df["drawdown"], "JPMC34 Drawdown", "jpmc34_drawdown.png", "Drawdown")
save_chart(df["volatility"], "JPMC34 Rolling Volatility", "jpmc34_volatility.png", "Volatility")

print("\nCharts saved:")
print(" - jpmc34_price.png")
print(" - jpmc34_drawdown.png")
print(" - jpmc34_volatility.png")

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

import yfinance as yf
import numpy as np
import pandas as pd

TICKER = "JPM"
PERIOD = "6mo"
TRADING_DAYS = 252

print(f"Downloading data for {TICKER}...")

ticker = yf.Ticker(TICKER)
df = ticker.history(period=PERIOD, auto_adjust=True)

if df.empty:
    raise ValueError("No data downloaded")

close = df["Close"].squeeze()
volume = df["Volume"].squeeze()

if len(close) < 60:
    raise ValueError("Not enough history (need >= 60 bars)")

returns = close.pct_change().dropna()

# -----------------------------
# CLASSICAL MARKET METRICS
# -----------------------------

last_price = float(close.iloc[-1])
momentum_3m = float(close.iloc[-1] / close.iloc[-60] - 1)
volatility = float(returns.std() * np.sqrt(TRADING_DAYS))

ma_50 = float(close.rolling(50).mean().iloc[-1])
trend_up = last_price > ma_50

last_volume = float(volume.iloc[-1])
avg_volume = float(volume.rolling(30).mean().iloc[-1])
liquidity_strong = last_volume > avg_volume

macro_stress = volatility > 0.25
corr_proxy = (abs(momentum_3m) > 0.05) and (volatility > 0.20)

print("\nCLASSICAL MARKET STATE")
print("-----------------------------------")
print(f"Last Price: {last_price:.2f} USD")
print(f"3M Momentum: {momentum_3m*100:.2f}%")
print(f"Volatility: {volatility*100:.2f}%")

# -----------------------------
# QUBIT ENCODING
# -----------------------------

q0 = int(momentum_3m < 0)
q1 = int(volatility > 0.22)
q2 = int(not trend_up)
q3 = int(not liquidity_strong)
q4 = int(macro_stress)
q5 = int(corr_proxy)

state = f"{q0}{q1}{q2}{q3}{q4}{q5}"

print("\nENCODED 6-QUBIT STATE")
print("-----------------------------------")
print("State:", state)

ones = state.count("1")
stress_probability = ones / 6

if ones <= 1:
    regime = "CALM / RISK-ON"
elif ones <= 3:
    regime = "NEUTRAL / MIXED"
elif ones <= 4:
    regime = "STRESS BUILDING"
else:
    regime = "HIGH STRESS / RISK-OFF"

print("\nREGIME DIAGNOSTIC")
print("-----------------------------------")
print(f"Active Stress Signals: {ones} / 6")
print(f"Stress Probability: {stress_probability*100:.2f}%")
print("Regime View:", regime)

# -----------------------------
# DECISION ENGINE
# -----------------------------

decision = "HOLD"

# Strong BUY Logic
if (q0 == 0 and     # Positive momentum
    q2 == 0 and     # Uptrend
    q4 == 0 and     # No macro stress
    q1 == 0):       # Volatility not high
    
    decision = "STRONG BUY"

# Strong SELL Logic
elif (q0 == 1 and
      q2 == 1 and
      (q4 == 1 or q1 == 1)):
    
    decision = "STRONG SELL"

print("\nTRADING DECISION")
print("-----------------------------------")
print("Action:", decision)

# -----------------------------
# JPMORGAN INFORMATION
# -----------------------------

info = ticker.info

print("\nJPMORGAN SNAPSHOT")
print("-----------------------------------")
print("Company:", info.get("longName"))
print("Sector:", info.get("sector"))
print("Market Cap:", info.get("marketCap"))
print("Trailing PE:", info.get("trailingPE"))
print("Forward PE:", info.get("forwardPE"))
print("Dividend Yield:", info.get("dividendYield"))

import yfinance as yf
import numpy as np
import pandas as pd

TICKER = "JPM"
PERIOD = "6mo"

print(f"Downloading data for {TICKER}...")

df = yf.download(TICKER, period=PERIOD)

close = df["Close"]
returns = close.pct_change().dropna()

# -----------------------------
# CLASSICAL MARKET METRICS
# -----------------------------

momentum_3m = close.iloc[-1] / close.iloc[-60] - 1
volatility = returns.std() * np.sqrt(252)

ma_50 = close.rolling(50).mean().iloc[-1]
trend_up = close.iloc[-1] > ma_50

volume = df["Volume"]
avg_volume = volume.rolling(30).mean().iloc[-1]
liquidity_strong = volume.iloc[-1] > avg_volume

# Macro / rates proxy (stylized)
macro_stress = volatility > 0.25

# Cross-asset correlation proxy (stylized)
corr_proxy = abs(momentum_3m) > 0.05 and volatility > 0.2

print("\nCLASSICAL MARKET STATE")
print("-----------------------------------")
print(f"Last Price: {close.iloc[-1]:.2f} USD")
print(f"3M Momentum: {momentum_3m*100:.2f}%")
print(f"Volatility: {volatility*100:.2f}%")

# -----------------------------
# QUBIT ENCODING
# -----------------------------

q0_momentum = int(momentum_3m < 0)          # 1 = negative momentum
q1_volatility = int(volatility > 0.22)       # 1 = high vol
q2_trend = int(not trend_up)                 # 1 = downtrend
q3_liquidity = int(not liquidity_strong)     # 1 = weak liquidity
q4_macro = int(macro_stress)                 # 1 = macro stress
q5_corr = int(corr_proxy)                    # 1 = crowded / correlated

state = f"{q0_momentum}{q1_volatility}{q2_trend}{q3_liquidity}{q4_macro}{q5_corr}"

print("\nENCODED 6-QUBIT STATE")
print("-----------------------------------")
print("State:", state)

# -----------------------------
# REGIME CLASSIFICATION LOGIC
# -----------------------------

ones = state.count("1")

if ones <= 1:
    regime = "CALM / RISK-ON"
elif ones <= 3:
    regime = "NEUTRAL / MIXED"
elif ones <= 4:
    regime = "STRESS BUILDING"
else:
    regime = "HIGH STRESS / RISK-OFF"

stress_probability = ones / 6

print("\nREGIME DIAGNOSTIC")
print("-----------------------------------")
print("Active Signals:", ones, "/ 6")
print(f"Stress Probability (stylized): {stress_probability*100:.2f}%")
print("Regime View:", regime)

import yfinance as yf
import numpy as np
import pandas as pd

TICKER = "JPM"
PERIOD = "6mo"
TRADING_DAYS = 252

print(f"Downloading data for {TICKER}...")

df = yf.download(TICKER, period=PERIOD, auto_adjust=True)

if df.empty:
    raise ValueError("No data downloaded")

close = df["Close"]
volume = df["Volume"]

if len(close) < 60:
    raise ValueError("Not enough history for regime model (need >= 60 bars)")

returns = close.pct_change().dropna()

# -----------------------------
# CLASSICAL MARKET METRICS (SCALAR SAFE)
# -----------------------------

last_price = float(close.iloc[-1])

momentum_3m = float(close.iloc[-1] / close.iloc[-60] - 1)

volatility = float(returns.std() * np.sqrt(TRADING_DAYS))

ma_50_series = close.rolling(50).mean()
ma_50 = float(ma_50_series.iloc[-1])

trend_up = last_price > ma_50

last_volume = float(volume.iloc[-1])
avg_volume_series = volume.rolling(30).mean()
avg_volume = float(avg_volume_series.iloc[-1])

liquidity_strong = last_volume > avg_volume

# Stylized macro / stress proxy
macro_stress = volatility > 0.25

# Stylized correlation / crowding proxy
corr_proxy = (abs(momentum_3m) > 0.05) and (volatility > 0.20)

print("\nCLASSICAL MARKET STATE")
print("-----------------------------------")
print(f"Last Price: {last_price:.2f} USD")
print(f"3M Momentum: {momentum_3m*100:.2f}%")
print(f"Volatility: {volatility*100:.2f}%")
print(f"Above MA50: {trend_up}")
print(f"Liquidity Strong: {liquidity_strong}")

# -----------------------------
# QUBIT ENCODING
# -----------------------------

q0_momentum = int(momentum_3m < 0)        # 1 = negative momentum
q1_volatility = int(volatility > 0.22)     # 1 = high vol
q2_trend = int(not trend_up)               # 1 = downtrend
q3_liquidity = int(not liquidity_strong)   # 1 = weak liquidity
q4_macro = int(macro_stress)               # 1 = macro stress
q5_corr = int(corr_proxy)                  # 1 = correlated / crowded

state = f"{q0_momentum}{q1_volatility}{q2_trend}{q3_liquidity}{q4_macro}{q5_corr}"

print("\nENCODED 6-QUBIT STATE")
print("-----------------------------------")
print("State:", state)

# -----------------------------
# REGIME CLASSIFICATION
# -----------------------------

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
print(f"Stress Probability (stylized): {stress_probability*100:.2f}%")
print("Regime View:", regime)

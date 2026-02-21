import yfinance as yf
import numpy as np
import pandas as pd

TICKER = "JPM"
BENCHMARK = "SPY"
PERIOD = "6mo"
TRADING_DAYS = 252

print(f"Downloading data for {TICKER}...")

ticker = yf.Ticker(TICKER)
bench = yf.Ticker(BENCHMARK)

df = ticker.history(period=PERIOD, auto_adjust=True)
df_bench = bench.history(period=PERIOD, auto_adjust=True)

if df.empty:
    raise ValueError("No data downloaded")

close = df["Close"]
volume = df["Volume"]
returns = close.pct_change().dropna()

# -----------------------------
# CLASSICAL METRICS
# -----------------------------

last_price = close.iloc[-1]
momentum_3m = close.iloc[-1] / close.iloc[-60] - 1
volatility = returns.std() * np.sqrt(TRADING_DAYS)

ma_50 = close.rolling(50).mean().iloc[-1]
trend_up = last_price > ma_50

last_volume = volume.iloc[-1]
avg_volume = volume.rolling(30).mean().iloc[-1]
liquidity_strong = last_volume > avg_volume

# Drawdown
rolling_max = close.cummax()
drawdown = (close / rolling_max - 1).iloc[-1]

# Relative Strength vs SPY
bench_close = df_bench["Close"]
rel_strength = (close / close.iloc[0]) - (bench_close / bench_close.iloc[0])

macro_stress = volatility > 0.25
corr_proxy = (abs(momentum_3m) > 0.05) and (volatility > 0.20)

# -----------------------------
# 8 QUBIT ENCODING
# -----------------------------

q0 = int(momentum_3m < 0)
q1 = int(volatility > 0.22)
q2 = int(not trend_up)
q3 = int(not liquidity_strong)
q4 = int(macro_stress)
q5 = int(corr_proxy)
q6 = int(drawdown < -0.10)
q7 = int(rel_strength.iloc[-1] < 0)

state = f"{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}"

print("\nENCODED 8-QUBIT STATE")
print("-----------------------------------")
print("State:", state)

# -----------------------------
# PROBABILISTIC REGIME SCORING
# -----------------------------

weights = np.array([1.2, 1.0, 1.2, 0.8, 1.1, 0.7, 1.3, 1.0])
signals = np.array([q0,q1,q2,q3,q4,q5,q6,q7])

stress_score = np.dot(weights, signals)
max_score = weights.sum()

stress_probability = stress_score / max_score

print("\nREGIME PROBABILITY")
print("-----------------------------------")
print(f"Stress Score: {stress_score:.2f} / {max_score:.2f}")
print(f"Stress Probability: {stress_probability*100:.2f}%")

# -----------------------------
# REGIME CLASSIFICATION
# -----------------------------

if stress_probability < 0.25:
    regime = "RISK-ON"
elif stress_probability < 0.50:
    regime = "NEUTRAL"
elif stress_probability < 0.70:
    regime = "STRESS BUILDING"
else:
    regime = "RISK-OFF"

print("Regime:", regime)

# -----------------------------
# DECISION ENGINE
# -----------------------------

decision = "HOLD"

if regime == "RISK-ON" and momentum_3m > 0 and trend_up:
    decision = "STRONG BUY"

elif regime == "RISK-OFF":
    decision = "STRONG SELL"

elif regime == "STRESS BUILDING":
    decision = "REDUCE EXPOSURE"

print("\nTRADING DECISION")
print("-----------------------------------")
print("Action:", decision)

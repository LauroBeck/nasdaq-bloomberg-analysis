import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional sklearn support (script still runs without it)
try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -----------------------------
# 1. Synthetic Bloomberg-like Data
# -----------------------------
np.random.seed(42)
n = 250

wti = np.cumsum(np.random.normal(0, 1, n)) + 70
brent = wti + np.random.normal(0, 0.8, n)
dxy = np.cumsum(np.random.normal(0, 0.2, n)) + 100

df = pd.DataFrame({
    "WTI": wti,
    "Brent": brent,
    "DXY": dxy
}).dropna()

# -----------------------------
# 2. Returns Calculation
# -----------------------------
returns = df.pct_change().dropna()

# -----------------------------
# 3. Regime Logic (Safe)
# -----------------------------
latest = returns.iloc[-1]

risk_on = False
risk_off = False

if latest["WTI"] > 0 and latest["DXY"] < 0:
    risk_on = True
elif latest["WTI"] < 0 and latest["DXY"] > 0:
    risk_off = True

print("\nLatest Market Snapshot")
print(latest)

if risk_on:
    print("\nRegime Detected: RISK ON")
elif risk_off:
    print("\nRegime Detected: RISK OFF")
else:
    print("\nRegime Detected: NEUTRAL")

# -----------------------------
# 4. Projection Engine
# -----------------------------
future_steps = 30
x = np.arange(len(df))

if SKLEARN_AVAILABLE:
    model = RandomForestRegressor(n_estimators=200, random_state=1)

    X = x.reshape(-1, 1)
    model.fit(X, df["WTI"])

    future_x = np.arange(len(df), len(df) + future_steps).reshape(-1, 1)
    projection = model.predict(future_x)

else:
    # Fallback projection (trend-based)
    trend = np.polyfit(x, df["WTI"], 1)
    projection = trend[0] * np.arange(len(df), len(df) + future_steps) + trend[1]

# -----------------------------
# 5. Plot Projection
# -----------------------------
plt.figure()
plt.plot(df.index, df["WTI"], label="WTI Historical")

future_index = np.arange(df.index[-1] + 1, df.index[-1] + 1 + future_steps)
plt.plot(future_index, projection, linestyle="--", label="WTI Projection")

plt.legend()
plt.title("WTI Projection Model")
plt.xlabel("Time")
plt.ylabel("Price")

plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional sklearn support (script still runs without it)
try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ==========================================
# 1. Synthetic Bloomberg-like Data
# ==========================================
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

df = df[(df > 0).all(axis=1)]

# ==========================================
# 2. Returns Calculation
# ==========================================
returns = df.pct_change().dropna()

latest = returns.iloc[-1]

# ==========================================
# 3. Regime Logic
# ==========================================
risk_on = False
risk_off = False

if latest["WTI"] > 0 and latest["DXY"] < 0:
    risk_on = True
elif latest["WTI"] < 0 and latest["DXY"] > 0:
    risk_off = True

print("\nLatest Market Snapshot")
print(latest)

if risk_on:
    regime_text = "RISK ON"
elif risk_off:
    regime_text = "RISK OFF"
else:
    regime_text = "NEUTRAL"

print(f"\nRegime Detected: {regime_text}")

# ==========================================
# 4. Projection Engine
# ==========================================
future_steps = 30
x = np.arange(len(df))

if SKLEARN_AVAILABLE:
    print("\nUsing RandomForestRegressor")

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=1
    )

    X = x.reshape(-1, 1)
    model.fit(X, df["WTI"])

    future_x = np.arange(len(df),
                         len(df) + future_steps).reshape(-1, 1)

    projection = model.predict(future_x)

else:
    print("\nSklearn not available → Using trend fallback")

    trend = np.polyfit(x, df["WTI"], 1)
    projection = trend[0] * np.arange(len(df),
                                     len(df) + future_steps) + trend[1]

# ==========================================
# 5. Bloomberg-Style Projection Plot
# ==========================================
plt.style.use("dark_background")

fig = plt.figure(figsize=(10, 6))

# Historical
plt.plot(df.index,
         df["WTI"],
         linewidth=2,
         label="WTI (Historical)")

# Future index
future_index = np.arange(df.index[-1] + 1,
                         df.index[-1] + 1 + future_steps)

# Projection
plt.plot(future_index,
         projection,
         linestyle="--",
         linewidth=2,
         label="WTI (Projection)")

# Confidence cone (synthetic uncertainty)
proj_std = np.std(df["WTI"].pct_change().dropna()) * df["WTI"].iloc[-1]

upper = projection + proj_std
lower = projection - proj_std

plt.fill_between(future_index,
                 lower,
                 upper,
                 alpha=0.15,
                 label="Projection Range")

# Last price marker
plt.scatter(df.index[-1],
            df["WTI"].iloc[-1],
            s=40,
            label="Last Price")

# Forward return estimate
expected_return = (projection[-1] / df["WTI"].iloc[-1] - 1) * 100

# Annotations
plt.text(0.02, 0.95,
         f"Regime: {regime_text}",
         transform=plt.gca().transAxes,
         fontsize=10,
         verticalalignment="top")

plt.text(0.02, 0.90,
         f"{future_steps}-Step Return: {expected_return:.2f}%",
         transform=plt.gca().transAxes,
         fontsize=10,
         verticalalignment="top")

# Styling
plt.title("WTI Projection Model", fontsize=14, pad=12)
plt.xlabel("Time")
plt.ylabel("Price")
plt.grid(alpha=0.2)
plt.legend()

output_file = "wti_projection_bloomberg.png"
plt.savefig(output_file, dpi=160, bbox_inches="tight")

print(f"\nProjection chart saved to {output_file}")
plt.close()

print("\nScript completed successfully!")

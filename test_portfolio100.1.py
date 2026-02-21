import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# =========================
# 1. LOAD & ALIGN DATA
# =========================

# Make sure indices are datetime
wti.index = pd.to_datetime(wti.index)
brent.index = pd.to_datetime(brent.index)
dxy.index = pd.to_datetime(dxy.index)

# Explicitly disable sorting (fix Pandas4Warning)
df = pd.concat([wti, brent, dxy], axis=1, sort=False)

df.columns = ["WTI", "Brent", "DXY"]

# Drop missing rows
df = df.dropna()

# Remove non-positive values before log
df = df[(df > 0).all(axis=1)]

# =========================
# 2. FEATURE ENGINEERING
# =========================

# Log returns (safe)
log_returns = np.log(df).diff()

# Rolling features
features = pd.DataFrame(index=df.index)
features["WTI_ret"] = log_returns["WTI"]
features["Brent_ret"] = log_returns["Brent"]
features["DXY_ret"] = log_returns["DXY"]

features["WTI_vol_5"] = log_returns["WTI"].rolling(5).std()
features["WTI_vol_20"] = log_returns["WTI"].rolling(20).std()

# Target = next day's WTI price (IMPORTANT FIX)
features["target"] = df["WTI"].shift(-1)

features = features.dropna()

# =========================
# 3. TRAIN / TEST SPLIT
# =========================

X = features.drop(columns="target")
y = features["target"]

# Time-series split (no shuffle)
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# =========================
# 4. TRAIN MODEL
# =========================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 5. PREDICT NEXT VALUE
# =========================

# Use last available feature row
X_last = X.iloc[[-1]]  # keep as DataFrame (fix sklearn warning)

predicted_price = model.predict(X_last)[0]
last_price = df["WTI"].iloc[-1]

print(f"Last WTI price: {last_price:.2f}")
print(f"Predicted next WTI price: {predicted_price:.2f}")

# =========================
# 6. PLOT RESULTS
# =========================

y_pred_test = model.predict(X_test)

plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, y_pred_test, label="Predicted")
plt.legend()
plt.title("WTI Prediction (Test Set)")
plt.tight_layout()

# Save instead of show (fix Agg warning)
plt.savefig("wti_prediction.png")
plt.close()

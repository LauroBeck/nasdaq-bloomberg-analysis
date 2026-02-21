import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# ===============================
# 1. DOWNLOAD DATA
# ===============================

# WTI Crude Oil Futures
wti = yf.download("CL=F", start="2015-01-01", progress=False)

# Brent Crude
brent = yf.download("BZ=F", start="2015-01-01", progress=False)

# Dollar Index
dxy = yf.download("DX-Y.NYB", start="2015-01-01", progress=False)

# Keep only Adjusted Close (or Close if Adj Close missing)
wti = wti[["Close"]].rename(columns={"Close": "WTI"})
brent = brent[["Close"]].rename(columns={"Close": "Brent"})
dxy = dxy[["Close"]].rename(columns={"Close": "DXY"})

# ===============================
# 2. ALIGN DATA (NO SORT WARNING)
# ===============================

df = pd.concat([wti, brent, dxy], axis=1, sort=False)
df = df.dropna()

# Remove non-positive values (prevents log warning)
df = df[(df > 0).all(axis=1)]

# ===============================
# 3. FEATURE ENGINEERING
# ===============================

log_returns = np.log(df).diff()

features = pd.DataFrame(index=df.index)

# Returns
features["WTI_ret"] = log_returns["WTI"]
features["Brent_ret"] = log_returns["Brent"]
features["DXY_ret"] = log_returns["DXY"]

# Volatility
features["WTI_vol_5"] = log_returns["WTI"].rolling(5).std()
features["WTI_vol_20"] = log_returns["WTI"].rolling(20).std()

# Target = next day's WTI price (NO LEAKAGE)
features["target"] = df["WTI"].shift(-1)

features = features.dropna()

# ===============================
# 4. TRAIN / TEST SPLIT
# ===============================

X = features.drop(columns="target")
y = features["target"]

split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# ===============================
# 5. TRAIN MODEL
# ===============================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 6. PREDICTION
# ===============================

# Predict test set
y_pred = model.predict(X_test)

# Predict next unseen value
X_last = X.iloc[[-1]]  # Keep as DataFrame (fix sklearn warning)
next_price_pred = model.predict(X_last)[0]
last_price = df["WTI"].iloc[-1]

print(f"Last WTI price: {last_price:.2f}")
print(f"Predicted next WTI price: {next_price_pred:.2f}")

# ===============================
# 7. PLOT RESULTS (NO GUI ERROR)
# ===============================

plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, y_pred, label="Predicted")
plt.title("WTI Next-Day Price Prediction")
plt.legend()
plt.tight_layout()

plt.savefig("wti_prediction.png")
plt.close()

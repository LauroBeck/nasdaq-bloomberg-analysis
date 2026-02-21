import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ===============================
# 1. BLOOMBERG PANEL TICKERS
# ===============================

tickers = {
    "WTI": "CL=F",
    "Brent": "BZ=F",
    "DXY": "DX-Y.NYB",
    "DOW": "^DJI",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "RBOB": "RB=F"
}

data = {}

for name, ticker in tickers.items():
    df = yf.download(ticker, start="2015-01-01", progress=False)
    data[name] = df["Close"]

df = pd.concat(data.values(), axis=1, sort=False)
df.columns = data.keys()
df = df.dropna()

# Remove non-positive values
df = df[(df > 0).all(axis=1)]

# ===============================
# 2. FEATURE ENGINEERING
# ===============================

log_returns = np.log(df).diff()

features = pd.DataFrame(index=df.index)

# Use all cross-asset returns as predictors
for col in df.columns:
    features[f"{col}_ret"] = log_returns[col]

# Volatility features
features["WTI_vol_5"] = log_returns["WTI"].rolling(5).std()
features["WTI_vol_20"] = log_returns["WTI"].rolling(20).std()

# Target = next-day WTI return (better than price)
features["target"] = log_returns["WTI"].shift(-1)

features = features.dropna()

# ===============================
# 3. TRAIN / TEST SPLIT
# ===============================

X = features.drop(columns="target")
y = features["target"]

split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# ===============================
# 4. TRAIN MODEL
# ===============================

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 5. PREDICTIONS
# ===============================

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE (Return Space): {rmse:.6f}")

# Convert predicted return to price
last_price = df["WTI"].iloc[-1]
next_return_pred = model.predict(X.iloc[[-1]])[0]
next_price_pred = last_price * np.exp(next_return_pred)

print(f"Last WTI price: {last_price:.2f}")
print(f"Predicted next WTI price: {next_price_pred:.2f}")

# ===============================
# 6. PLOT RESULTS
# ===============================

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Actual Return")
plt.plot(y_test.index, y_pred, label="Predicted Return")
plt.title("Bloomberg Cross-Asset Model – WTI Next-Day Return")
plt.legend()
plt.tight_layout()
plt.savefig("bloomberg_style_wti_model.png")
plt.close()

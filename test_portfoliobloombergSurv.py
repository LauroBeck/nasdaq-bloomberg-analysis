import matplotlib
matplotlib.use("Agg")   # ✅ Safe for Docker / servers / headless

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# ==========================================
# 1. DOWNLOAD DATA
# ==========================================
symbol = "CL=F"   # WTI NYMEX Futures

data = yf.download(symbol, start="2020-01-01", progress=False)

# yfinance sometimes returns multi-index columns → normalize
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

df = data[["Close"]].copy()
df = df.dropna()
df = df[df["Close"] > 0]

# ==========================================
# 2. FEATURE ENGINEERING (LAGS)
# ==========================================
df["Lag_1"] = df["Close"].shift(1)
df["Lag_2"] = df["Close"].shift(2)
df["Lag_3"] = df["Close"].shift(3)

df.dropna(inplace=True)

X = df[["Lag_1", "Lag_2", "Lag_3"]]
y = df["Close"]

# ==========================================
# 3. TIME-SERIES SAFE SPLIT
# ==========================================
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

# ==========================================
# 4. MODEL
# ==========================================
model = LinearRegression()
model.fit(X_train, y_train)

df["Prediction"] = model.predict(X)

# ==========================================
# 5. SAFE SCALAR EXTRACTION (CRITICAL FIX)
# ==========================================
last_price = float(df["Close"].iloc[-1])

last_features = X.iloc[-1].values.reshape(1, -1)
next_price = float(model.predict(last_features)[0])

# ==========================================
# 6. BLOOMBERG-STYLE CHART
# ==========================================
plt.style.use("dark_background")

plt.figure(figsize=(10, 5))

plt.plot(df.index, df["Close"], label="Actual Price", linewidth=2)
plt.plot(df.index, df["Prediction"], label="Model Prediction", linestyle="--")

plt.scatter(df.index[-1], last_price, s=40, label="Last Price")

plt.title("WTI NYMEX Price Prediction (Linear Regression)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(alpha=0.2)
plt.legend()
plt.tight_layout()

output_file = "wti_prediction.png"
plt.savefig(output_file, dpi=160, bbox_inches="tight")
plt.close()

# ==========================================
# 7. OUTPUT
# ==========================================
print(f"Chart saved to {output_file}")
print(f"Last WTI price: {last_price:.2f} USD")
print(f"Predicted next WTI price: {next_price:.2f} USD")
print("Script completed successfully!")

import matplotlib
matplotlib.use("Agg")   # ✅ CRITICAL → disables GUI everywhere

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

df = data[['Close']].copy()
df = df.dropna()
df = df[df['Close'] > 0]

# ==========================================
# 2. FEATURE ENGINEERING (LAGS)
# ==========================================
df['Lag_1'] = df['Close'].shift(1)
df['Lag_2'] = df['Close'].shift(2)
df['Lag_3'] = df['Close'].shift(3)

df.dropna(inplace=True)

X = df[['Lag_1', 'Lag_2', 'Lag_3']]
y = df['Close']

# ==========================================
# 3. TIME-SERIES SAFE SPLIT
# ==========================================
split_index = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# ==========================================
# 4. MODEL
# ==========================================
model = LinearRegression()
model.fit(X_train, y_train)

df['Prediction'] = model.predict(X)

# ==========================================
# 5. SAFE PLOT (NO GUI REQUIRED)
# ==========================================
plt.style.use("dark_background")

plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Close'], label='Actual Price', linewidth=2)
plt.plot(df.index, df['Prediction'], label='Predicted Price', linestyle='--')

plt.legend()
plt.title("WTI NYMEX Price Prediction (Linear Regression)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(alpha=0.2)
plt.tight_layout()

output_file = "wti_linear_regression.png"
plt.savefig(output_file, dpi=160, bbox_inches="tight")
plt.close()

print(f"Chart saved to {output_file}")

# ==========================================
# 6. NEXT PRICE FORECAST
# ==========================================
last_values = X.iloc[-1].values.reshape(1, -1)
next_price = model.predict(last_values)

print(f"Last WTI price: {df['Close'].iloc[-1]:.2f} USD")
print(f"Predicted next WTI price: {next_price[0]:.2f} USD")

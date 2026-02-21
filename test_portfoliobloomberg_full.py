import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ==========================================
# 1. CONFIGURATION
# ==========================================
START_DATE = "2015-01-01"

TICKERS = {
    "WTI": "CL=F",
    "BRENT": "BZ=F",
    "DXY": "DX-Y.NYB",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW": "^DJI",
    "RBOB": "RB=F"
}

# Map raw tickers to clean column names
TICKER_NAMES = {
    "CL=F": "WTI",
    "BZ=F": "BRENT",
    "DX-Y.NYB": "DXY",
    "^GSPC": "SP500",
    "^IXIC": "NASDAQ",
    "^DJI": "DOW",
    "RB=F": "RBOB"
}

# ==========================================
# 2. SAFE DOWNLOAD FUNCTION
# ==========================================
def safe_download(name, ticker):
    try:
        df = yf.download(ticker, start=START_DATE, progress=False)
        if df.empty:
            print(f"WARNING: No data for {name}")
            return None
        close = df["Close"].copy()
        close.name = ticker  # keep raw ticker for now
        return close
    except Exception as e:
        print(f"ERROR downloading {name}: {e}")
        return None

# ==========================================
# 3. DOWNLOAD DATA
# ==========================================
series_list = []
for name, ticker in TICKERS.items():
    s = safe_download(name, ticker)
    if s is not None:
        series_list.append(s)

df = pd.concat(series_list, axis=1, sort=False)
df = df.dropna()
df = df[(df > 0).all(axis=1)]

# Rename columns to clean names
df.rename(columns=TICKER_NAMES, inplace=True)
print("Columns after rename:", df.columns.tolist())
print("Dataset shape:", df.shape)

# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================
log_returns = np.log(df).diff()

features = pd.DataFrame(index=df.index)

# Add returns for all assets
for col in df.columns:
    features[f"{col}_RET"] = log_returns[col]

# WTI volatility features
features["WTI_VOL_5"] = log_returns["WTI"].rolling(5).std()
features["WTI_VOL_20"] = log_returns["WTI"].rolling(20).std()

# Spread feature
features["SPREAD_WTI_BRENT"] = df["WTI"] - df["BRENT"]

# Target = next-day WTI return
features["TARGET"] = log_returns["WTI"].shift(-1)

# Drop NaNs
features = features.dropna()
print("Features shape:", features.shape)
print(features.head())

# ==========================================
# 5. PLOT PRICES
# ==========================================
plt.figure(figsize=(12,6))
plt.plot(df.index, df["WTI"], label="WTI")
plt.plot(df.index, df["BRENT"], label="Brent")
plt.title("WTI vs Brent Prices")
plt.legend()
plt.tight_layout()
plt.savefig("wti_vs_brent.png")
plt.close()

# Plot target
plt.figure(figsize=(12,6))
plt.plot(features.index, features["TARGET"], label="WTI Next-Day Return")
plt.title("WTI Next-Day Return")
plt.legend()
plt.tight_layout()
plt.savefig("wti_nextday_return.png")
plt.close()

# ==========================================
# 6. TRAIN / TEST SPLIT
# ==========================================
X = features.drop(columns="TARGET")
y = features["TARGET"]

split_idx = int(len(X)*0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ==========================================
# 7. TRAIN RANDOM FOREST MODEL
# ==========================================
model = RandomForestRegressor(n_estimators=400, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE (return space): {rmse:.6f}")

# Predict next-day WTI price
X_last = X.iloc[[-1]]
next_return = model.predict(X_last)[0]
last_price = df["WTI"].iloc[-1]
next_price = last_price * np.exp(next_return)

print(f"Last WTI price: {last_price:.2f}")
print(f"Predicted next WTI price: {next_price:.2f}")

print("Script completed successfully!")

import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------------------------------
# Dependency check (scikit-learn)
# ---------------------------------
try:
    from sklearn.ensemble import RandomForestRegressor
except ModuleNotFoundError:
    print("\nERROR: scikit-learn is not installed.\n")
    print("Install it with:\n")
    print("  pip install scikit-learn\n")
    sys.exit()

# ---------------------------------
# Safe downloader
# ---------------------------------
def safe_download(symbol):
    print(f"Downloading {symbol} ...")
    data = yf.download(symbol, start="2020-01-01", progress=False)

    if data is None or data.empty:
        raise RuntimeError(f"Download failed for {symbol}")

    if 'Close' not in data.columns:
        raise RuntimeError(f"'Close' column missing for {symbol}")

    return data['Close']

# ---------------------------------
# Data acquisition with fallback
# ---------------------------------
try:
    wti = safe_download("CL=F")      # WTI Futures
    brent = safe_download("BZ=F")    # Brent Futures

    # DXY is fragile → fallback logic
    try:
        dxy = safe_download("DX-Y.NYB")
    except Exception:
        print("Primary DXY failed → using DX=F fallback")
        dxy = safe_download("DX=F")

    print("\nDownload successful\n")

    print("WTI rows :", len(wti))
    print("Brent rows:", len(brent))
    print("DXY rows  :", len(dxy))

    df = pd.concat([wti, brent, dxy], axis=1)
    df.columns = ['WTI', 'Brent', 'DXY']
    df.dropna(inplace=True)

except Exception as e:
    print("\nDATA ERROR:", str(e))
    sys.exit()

# ---------------------------------
# Returns (log returns)
# ---------------------------------
returns = np.log(df / df.shift(1))

df['WTI_ret'] = returns['WTI']
df['Brent_ret'] = returns['Brent']
df['DXY_ret'] = returns['DXY']

# ---------------------------------
# Feature Engineering
# ---------------------------------
df['volatility'] = df['WTI_ret'].rolling(10).std()
df['momentum'] = df['WTI_ret'].rolling(5).mean()

df['lag1'] = df['WTI_ret'].shift(1)
df['lag2'] = df['WTI_ret'].shift(2)
df['lag3'] = df['WTI_ret'].shift(3)

# Target = next-day return
df['target'] = df['WTI_ret'].shift(-1)

df.dropna(inplace=True)

features = [
    'lag1', 'lag2', 'lag3',
    'volatility', 'momentum',
    'Brent_ret', 'DXY_ret'
]

X = df[features]
y = df['target']

# ---------------------------------
# Time-series safe split
# ---------------------------------
split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---------------------------------
# Model
# ---------------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

df['predicted_ret'] = model.predict(X)

# Convert return → price
df['predicted_price'] = df['WTI'].shift(1) * np.exp(df['predicted_ret'])

# ---------------------------------
# Plot
# ---------------------------------
plt.figure()
plt.plot(df.index, df['WTI'], label="Actual WTI")
plt.plot(df.index, df['predicted_price'], label="Predicted WTI")
plt.legend()
plt.title("WTI NYMEX Prediction Model")
plt.show()

# ---------------------------------
# Next-day Forecast
# ---------------------------------
last_row = X.iloc[-1].values.reshape(1, -1)
next_ret = model.predict(last_row)[0]

last_price = df['WTI'].iloc[-1]
next_price = last_price * np.exp(next_ret)

print(f"\nLast WTI price: {last_price:.2f}")
print(f"Predicted next WTI price: {next_price:.2f}\n")

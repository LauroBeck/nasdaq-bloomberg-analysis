import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ==========================================
# 1. CONFIGURATION
# ==========================================
START_DATE = "2015-01-01"

TICKERS = {
    "WTI": "CL=F",       # WTI Crude
    "Brent": "BZ=F",     # Brent Crude
    "DXY": "DX-Y.NYB",   # US Dollar Index
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW": "^DJI",
    "RBOB": "RB=F"
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
        # Keep only Close column and rename
        close = df["Close"].copy()
        close.name = name.upper()
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

# Concatenate into one DataFrame
df = pd.concat(series_list, axis=1, sort=False)

# Drop missing values and remove non-positive prices
df = df.dropna()
df = df[(df > 0).all(axis=1)]

# Uppercase & strip column names to avoid KeyErrors
df.columns = [col.strip().upper() for col in df.columns]

print("Columns in df:", df.columns.tolist())
print("Dataset shape:", df.shape)

# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================
log_returns = np.log(df).diff()

features = pd.DataFrame(index=df.index)

# Add returns for all assets
for col in df.columns:
    features[f"{col}_RET"] = log_returns[col]

# Volatility features for WTI
features["WTI_VOL_5"] = log_returns["WTI"].rolling(5).std()
features["WTI_VOL_20"] = log_returns["WTI"].rolling(20).std()

# Spread feature
features["SPREAD_WTI_BRENT"] = df["WTI"] - df["BRENT"]

# Target = next-day WTI return (stationary)
features["TARGET"] = log_returns["WTI"].shift(-1)

# Drop NaNs after rolling and shift
features = features.dropna()

print("Features shape:", features.shape)
print(features.head())

# ==========================================
# 5. QUICK PLOTS
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["WTI"], label="WTI")
plt.plot(df.index, df["BRENT"], label="Brent")
plt.title("WTI vs Brent Prices")
plt.legend()
plt.tight_layout()
plt.savefig("wti_brent_prices.png")
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(features.index, features["TARGET"], label="WTI Next-Day Return")
plt.title("WTI Next-Day Return (for Modeling)")
plt.legend()
plt.tight_layout()
plt.savefig("wti_nextday_return.png")
plt.close()

print("Script completed successfully!")

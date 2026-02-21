import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ==========================================
# 1. CONFIGURATION
# ==========================================

START_DATE = "2015-01-01"

TICKERS = {
    "WTI": "CL=F",
    "Brent": "BZ=F",
    "DXY": "DX-Y.NYB",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW": "^DJI",
    "RBOB": "RB=F"
}

# ==========================================
# 2. SAFE DOWNLOAD (CLOSE ONLY)
# ==========================================

def safe_download(name, ticker):
    try:
        df = yf.download(ticker, start=START_DATE, progress=False)

        if df.empty:
            print(f"WARNING: No data for {name}")
            return None

        close = df["Close"].copy()
        close.name = name

        return close

    except Exception as e:
        print(f"ERROR downloading {name}: {e}")
        return None


# ==========================================
# 3. DOWNLOAD & BUILD PANEL
# ==========================================

series_list = []

for name, ticker in TICKERS.items():
    s = safe_download(name, ticker)
    if s is not None:
        series_list.append(s)

df = pd.concat(series_list, axis=1, sort=False)

# Drop missing values
df = df.dropna()

# Remove non-positive values (prevents log errors)
df = df[(df > 0).all(axis=1)]

print("Final dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

# Log returns
log_returns = np.log(df).diff()

features = pd.DataFrame(index=df.index)

# Add returns for all assets
for col in df.columns:
    features[f"{col}_ret"] = log_returns[col]

# Add WTI volatility
features["WTI_vol_5"] = log_returns["WTI"].rolling(5).std()
features["WTI_vol_20"] = log_returns["WTI"].rolling(20).std()

# Add spread (THIS NOW WORKS)
features["Spread_WTI_Brent"] = df["WTI"] - df["Brent"]

# Target = next-day WTI return
features["target"] = log_returns["WTI"].shift(-1)

features = features.dropna()

print("Feature set shape:", features.shape)

# ==========================================
# 5. QUICK VISUAL CHECK
# ==========================================

plt.figure(figsize=(12, 6))
plt.plot(df.index, df["WTI"], label="WTI")
plt.plot(df.index, df["Brent"], label="Brent")
plt.title("WTI vs Brent")
plt.legend()
plt.tight_layout()
plt.savefig("bloomberg_panel_prices.png")
plt.close()

print("Script completed successfully.")

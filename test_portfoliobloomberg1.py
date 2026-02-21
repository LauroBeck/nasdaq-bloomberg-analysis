import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ==========================================
# 1. CONFIGURATION
# ==========================================

START_DATE = "2015-01-01"

TICKERS = {
    "WTI": "CL=F",        # WTI Crude
    "Brent": "BZ=F",      # Brent Crude
    "DXY": "DX-Y.NYB"     # Dollar Index
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

        if "Close" not in df.columns:
            print(f"WARNING: Close column missing for {name}")
            return None

        df = df[["Close"]].rename(columns={"Close": name})
        df = df[df[name] > 0]  # remove invalid prices

        return df

    except Exception as e:
        print(f"DOWNLOAD ERROR [{name}]: {e}")
        return None

# ==========================================
# 3. DOWNLOAD DATA
# ==========================================

frames = []

for name, ticker in TICKERS.items():
    df_tmp = safe_download(name, ticker)
    if df_tmp is not None:
        frames.append(df_tmp)

if len(frames) < 2:
    raise RuntimeError("Not enough market series downloaded.")

df = pd.concat(frames, axis=1, sort=False).dropna()

# ==========================================
# 4. RETURNS & VOLATILITY
# ==========================================

log_returns = np.log(df).diff().dropna()

if log_returns.empty:
    raise RuntimeError("Return calculation failed.")

vol_5 = log_returns.rolling(5).std()
vol_20 = log_returns.rolling(20).std()

# ==========================================
# 5. CROSS-MARKET METRICS
# ==========================================

corr_matrix = log_returns.corr()

def compute_beta(y, x):
    try:
        cov = np.cov(y, x)
        return float(cov[0, 1] / cov[1, 1])
    except Exception:
        return np.nan

beta_wti_dxy = compute_beta(
    log_returns["WTI"],
    log_returns["DXY"]
) if "WTI" in log_returns and "DXY" in log_returns else np.nan

# Spread (very Bloomberg-like metric)
df["Spread_WTI_Brent"] = df["WTI"] - df["Brent"]

# ==========================================
# 6. SIGNAL ENGINE (DEFENSIVE)
# ==========================================

latest = log_returns.iloc[-1]

wti_ret = float(latest.get("WTI", np.nan))
dxy_ret = float(latest.get("DXY", np.nan))

signal = "NEUTRAL"

if np.isfinite(wti_ret) and np.isfinite(dxy_ret):

    if wti_ret > 0 and dxy_ret < 0:
        signal = "RISK-ON (Oil ↑ / Dollar ↓)"

    elif wti_ret < 0 and dxy_ret > 0:
        signal = "RISK-OFF (Oil ↓ / Dollar ↑)"

    elif wti_ret > 0:
        signal = "WTI MOMENTUM POSITIVE"

    elif wti_ret < 0:
        signal = "WTI MOMENTUM NEGATIVE"

else:
    signal = "INSUFFICIENT DATA"

# ==========================================
# 7. SAFE NUMERIC SNAPSHOT
# ==========================================

def safe_last(series):
    try:
        return float(series.dropna().iloc[-1])
    except Exception:
        return np.nan

last_price = safe_last(df["WTI"])
last_vol5 = safe_last(vol_5["WTI"])
last_vol20 = safe_last(vol_20["WTI"])

print("\n===== MARKET SNAPSHOT =====")

if np.isfinite(last_price):
    print(f"Last WTI Price: {last_price:.2f}")
else:
    print("Last WTI Price: N/A")

if np.isfinite(last_vol5):
    print(f"WTI Vol (5d): {last_vol5:.4f}")

if np.isfinite(last_vol20):
    print(f"WTI Vol (20d): {last_vol20:.4f}")

if np.isfinite(beta_wti_dxy):
    print(f"WTI vs DXY Beta: {beta_wti_dxy:.3f}")

print(f"Signal: {signal}")

print("\n===== CORRELATION MATRIX =====")
print(corr_matrix)

# ==========================================
# 8. CHARTS (HEADLESS SAFE)
# ==========================================

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["WTI"], label="WTI")
plt.plot(df.index, df["Brent"], label="Brent")
plt.title("Crude Oil Futures")
plt.legend()
plt.tight_layout()
plt.savefig("oil_prices.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(vol_20.index, vol_20["WTI"], label="WTI 20d Vol")
plt.title("WTI Volatility Regime")
plt.legend()
plt.tight_layout()
plt.savefig("wti_volatility.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["Spread_WTI_Brent"], label="WTI-Brent Spread")
plt.title("WTI vs Brent Spread")
plt.legend()
plt.tight_layout()
plt.savefig("spread.png")
plt.close()

print("\nCharts saved:")
print(" - oil_prices.png")
print(" - wti_volatility.png")
print(" - spread.png")

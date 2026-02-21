import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ==========================================
# 1. DOWNLOAD MARKET DATA
# ==========================================

symbols = {
    "WTI": "CL=F",
    "Brent": "BZ=F",
    "DXY": "DX-Y.NYB"
}

data = {}

for name, ticker in symbols.items():
    df = yf.download(ticker, start="2015-01-01", progress=False)
    df = df[["Close"]].rename(columns={"Close": name})
    data[name] = df

# Align safely
df = pd.concat(data.values(), axis=1, sort=False).dropna()

# Remove invalid values
df = df[(df > 0).all(axis=1)]

# ==========================================
# 2. RETURNS & VOLATILITY
# ==========================================

log_returns = np.log(df).diff().dropna()

vol_5 = log_returns.rolling(5).std()
vol_20 = log_returns.rolling(20).std()

# ==========================================
# 3. CROSS-MARKET STRUCTURE
# ==========================================

corr_matrix = log_returns.corr()

# Beta of WTI vs DXY (simple OLS approximation)
cov = np.cov(log_returns["WTI"], log_returns["DXY"])
beta_wti_dxy = cov[0, 1] / cov[1, 1]

# ==========================================
# 4. REGIME / SIGNAL LOGIC
# ==========================================

latest = log_returns.iloc[-1]

signal = "NEUTRAL"

if latest["WTI"] > 0 and latest["DXY"] < 0:
    signal = "RISK-ON (Oil supportive)"
elif latest["WTI"] < 0 and latest["DXY"] > 0:
    signal = "RISK-OFF (Dollar pressure)"
elif latest["WTI"] > 0:
    signal = "WTI MOMENTUM POSITIVE"
elif latest["WTI"] < 0:
    signal = "WTI MOMENTUM NEGATIVE"

# ==========================================
# 5. SAFE NUMERIC FORMATTING
# ==========================================

last_price = float(df["WTI"].iloc[-1])
last_vol5 = float(vol_5["WTI"].iloc[-1])
last_vol20 = float(vol_20["WTI"].iloc[-1])
beta_wti_dxy = float(beta_wti_dxy)

print("\n===== MARKET SNAPSHOT =====")
print(f"Last WTI Price: {last_price:.2f}")
print(f"WTI Vol (5d): {last_vol5:.4f}")
print(f"WTI Vol (20d): {last_vol20:.4f}")
print(f"WTI vs DXY Beta: {beta_wti_dxy:.3f}")
print(f"Signal: {signal}")

print("\n===== CORRELATION MATRIX =====")
print(corr_matrix)

# ==========================================
# 6. VISUALIZATION
# ==========================================

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["WTI"], label="WTI")
plt.plot(df.index, df["Brent"], label="Brent")
plt.title("Crude Oil Futures Structure")
plt.legend()
plt.tight_layout()
plt.savefig("oil_structure.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(vol_20.index, vol_20["WTI"], label="WTI 20d Vol")
plt.title("WTI Volatility Regime")
plt.legend()
plt.tight_layout()
plt.savefig("wti_volatility.png")
plt.close()

print("\nCharts saved:")
print(" - oil_structure.png")
print(" - wti_volatility.png")

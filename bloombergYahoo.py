# ==========================================================
# FGKFX PROFESSIONAL PORTFOLIO ENGINE (Yahoo Finance Edition)
# Replicating Bloomberg Terminal: Fidelity Growth Co K6 Fund
# ==========================================================

import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.style.use("dark_background")

# ----------------------------------------------------------
# CONFIGURATION (Mapped from Bloomberg Image)
# ----------------------------------------------------------
# Mapping Bloomberg names to YFinance Tickers
ticker_map = {
    "NVDA": "NVIDIA CORP",
    "AAPL": "APPLE INC",
    "MSFT": "MICROSOFT CORP",
    "AVGO": "BROADCOM INC",
    "ORCL": "ORACLE CORP",
    "PSTG": "PURE STORAGE INC",
    "CRM":  "SALESFORCE INC",
    "NET":  "CLOUDFLARE INC",
    "NOW":  "SERVICENOW INC",
    "AMD":  "ADVANCED MICRO DEVICES"
}
tickers = list(ticker_map.keys())

# Setup matching the 3-year lookback in the image title
start_date = "2021-01-01"
risk_free_rate = 0.04
trading_days = 252
rebalance_frequency = 63  
rolling_window = 252  

# Volatility caps (Ensuring the engine doesn't over-rely on NVDA's 879% run)
vol_caps = {t: 0.15 for t in tickers}
vol_caps["NVDA"] = 0.20 # Allowing higher weight for the top contributor

# ----------------------------------------------------------
# DOWNLOAD DATA
# ----------------------------------------------------------
print(f"Fetching data for FGKFX holdings: {tickers}...")
prices = yf.download(tickers, start=start_date, auto_adjust=True)["Close"].dropna()
returns = prices.pct_change().dropna()

# ----------------------------------------------------------
# OPTIMIZATION FUNCTIONS
# ----------------------------------------------------------
def get_perf(w, r, c):
    ret = np.dot(w, r.mean() * trading_days)
    vol = np.sqrt(np.dot(w.T, np.dot(c.cov() * trading_days, w)))
    return ret, vol, (ret - risk_free_rate) / vol

def neg_sharpe(w, r, c): return -get_perf(w, r, c)[2]
def min_vol(w, r, c): return get_perf(w, r, c)[1]

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
init_w = np.array(len(tickers) * [1./len(tickers)])

# ----------------------------------------------------------
# ROLLING BACKTEST
# ----------------------------------------------------------
results_max_s, results_min_v, dates = [], [], []

for i in range(rolling_window, len(returns) - rebalance_frequency, rebalance_frequency):
    window = returns.iloc[i-rolling_window:i]
    
    # Max Sharpe with Vol Caps
    bnds_cap = tuple((0, vol_caps[t]) for t in tickers)
    res_s = minimize(neg_sharpe, init_w, args=(window, window), 
                     method='SLSQP', bounds=bnds_cap, constraints=cons)
    
    # Min Volatility
    res_v = minimize(min_vol, init_w, args=(window, window), 
                     method='SLSQP', bounds=tuple((0,1) for _ in tickers), constraints=cons)

    nxt_ret = returns.iloc[i:i+rebalance_frequency]
    results_max_s.extend(nxt_ret @ res_s.x)
    results_min_v.extend(nxt_ret @ res_v.x)
    dates.extend(nxt_ret.index)

# ----------------------------------------------------------
# VISUALIZATION & ANALYSIS
# ----------------------------------------------------------
s_series = pd.Series(results_max_s, index=dates)
v_series = pd.Series(results_min_v, index=dates)

plt.figure(figsize=(12,6))
plt.plot((1 + s_series).cumprod(), label="Strategy: Vol-Capped Max Sharpe", color="lime")
plt.plot((1 + v_series).cumprod(), label="Strategy: Min Volatility", color="cyan")
plt.title("FGKFX Replication: Dual Strategy Returns")
plt.ylabel("Cumulative Growth")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

print("\n===== FINAL METRICS (REPLICATING BLOOMBERG DATA) =====")
for name, ser in [("MAX SHARPE", s_series), ("MIN VOL", v_series)]:
    ann_ret = (1 + ser.mean())**trading_days - 1
    ann_vol = ser.std() * np.sqrt(trading_days)
    print(f"{name} -> Return: {ann_ret:.2%}, Vol: {ann_vol:.2%}, Sharpe: {(ann_ret-risk_free_rate)/ann_vol:.2f}")

print("\n=== SUCCESS: ENGINE DEPLOYED ON UBUNTU ===")

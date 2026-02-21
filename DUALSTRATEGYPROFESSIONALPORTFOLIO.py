# ==========================================================
# DUAL-STRATEGY PROFESSIONAL PORTFOLIO ENGINE: FGKFX EDITION
# Assets: NVDA, AAPL, MSFT, AVGO, ORCL, CRM, AMD
# Features:
# 1. Volatility-capped rolling Max Sharpe
# 2. Rolling Min Volatility
# 3. Combined cumulative return plots
# 4. Rolling Sharpe / volatility metrics
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
# CONFIGURATION (Based on FGKFX Top Tech Holdings)
# ----------------------------------------------------------
tickers = ["NVDA", "AAPL", "MSFT", "AVGO", "ORCL", "CRM", "AMD"]
start_date = "2019-01-01"  # Capture the 3-5 year tech run
risk_free_rate = 0.04      # Adjusted for current interest rate environment
trading_days = 252
rebalance_frequency = 63   # Quarterly rebalance
rolling_window = 252       # 1-year rolling lookback

# Volatility caps based on current portfolio weight distribution in image
vol_caps = {
    "NVDA": 0.20, "AAPL": 0.15, "MSFT": 0.15, 
    "AVGO": 0.12, "ORCL": 0.10, "CRM": 0.10, "AMD": 0.10
}

# ----------------------------------------------------------
# DOWNLOAD DATA
# ----------------------------------------------------------
print(f"Downloading data for: {tickers}...")
raw = yf.download(tickers, start=start_date, auto_adjust=True)

if isinstance(raw.columns, pd.MultiIndex):
    prices = raw["Close"]
else:
    prices = raw[["Close"]]

prices = prices.dropna()
returns = prices.pct_change(fill_method=None).dropna()

# ----------------------------------------------------------
# STATIC PORTFOLIO CALCULATIONS
# ----------------------------------------------------------
mean_returns = returns.mean() * trading_days
cov_matrix = returns.cov() * trading_days

def portfolio_performance(weights, mean_ret=None, cov=None):
    if mean_ret is None: mean_ret = mean_returns
    if cov is None: cov = cov_matrix
    
    ret = np.dot(weights, mean_ret)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    sharpe = (ret - risk_free_rate) / vol
    return ret, vol, sharpe

# Optimization Functions
def negative_sharpe(weights, mean_ret, cov):
    return -portfolio_performance(weights, mean_ret, cov)[2]

def portfolio_volatility(weights, mean_ret, cov):
    return portfolio_performance(weights, mean_ret, cov)[1]

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(tickers)))
initial_weights = np.array(len(tickers) * [1./len(tickers)])

# Static Max Sharpe
opt_sharpe = minimize(negative_sharpe, initial_weights, args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
max_sharpe_weights = opt_sharpe.x
max_sharpe_perf = portfolio_performance(max_sharpe_weights)

# Static Min Volatility
opt_vol = minimize(portfolio_volatility, initial_weights, args=(mean_returns, cov_matrix),
                   method='SLSQP', bounds=bounds, constraints=constraints)
min_vol_weights = opt_vol.x
min_vol_perf = portfolio_performance(min_vol_weights)

# ----------------------------------------------------------
# EFFICIENT FRONTIER PLOT
# ----------------------------------------------------------
num_portfolios = 5000
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    w = np.random.random(len(tickers))
    w /= np.sum(w)
    r, v, s = portfolio_performance(w)
    results[0,i] = r
    results[1,i] = v
    results[2,i] = s

plt.figure(figsize=(10,6))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', alpha=0.5)
plt.colorbar(label="Sharpe Ratio")
plt.scatter(max_sharpe_perf[1], max_sharpe_perf[0], color='red', marker='*', s=300, label="Max Sharpe")
plt.scatter(min_vol_perf[1], min_vol_perf[0], color='cyan', marker='D', s=150, label="Min Volatility")
plt.xlabel("Annualized Volatility")
plt.ylabel("Annualized Return")
plt.title("Efficient Frontier: FGKFX Technology Components")
plt.legend()
plt.show()

# ----------------------------------------------------------
# ROLLING STRATEGY BACKTEST
# ----------------------------------------------------------
rolling_max_sharpe_values = []
rolling_min_vol_values = []
dates = []

for i in range(rolling_window + rebalance_frequency, len(returns), rebalance_frequency):
    window_returns = returns.iloc[i-rolling_window:i]
    if window_returns.shape[0] < rolling_window: continue

    mean_r = window_returns.mean() * trading_days
    cov_m = window_returns.cov() * trading_days

    # Volatility-capped Max Sharpe optimization
    bounds_cap = tuple((0, vol_caps[t]) for t in tickers)
    opt_cap = minimize(negative_sharpe, initial_weights, args=(mean_r, cov_m),
                       method='SLSQP', bounds=bounds_cap, constraints=constraints)
    
    # Min Volatility optimization
    opt_min = minimize(portfolio_volatility, initial_weights, args=(mean_r, cov_m),
                       method='SLSQP', bounds=bounds, constraints=constraints)

    period_returns = returns.iloc[i:i+rebalance_frequency]
    rolling_max_sharpe_values.extend(period_returns @ opt_cap.x)
    rolling_min_vol_values.extend(period_returns @ opt_min.x)
    dates.extend(period_returns.index)

rolling_max_series = pd.Series(rolling_max_sharpe_values, index=dates)
rolling_min_series = pd.Series(rolling_min_vol_values, index=dates)

# ----------------------------------------------------------
# PLOT RESULTS & PRINT METRICS
# ----------------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot((1 + rolling_max_series).cumprod(), label="Rolling Max Sharpe (Capped)", color="#FF3333", linewidth=2)
plt.plot((1 + rolling_min_series).cumprod(), label="Rolling Min Volatility", color="#33CCFF", linewidth=2)
plt.title("FGKFX Strategy Backtest: Cumulative Returns")
plt.ylabel("Growth of $1")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

def compute_metrics(series, label):
    total_ret = (1 + series).prod() - 1
    ann_vol = series.std() * np.sqrt(trading_days)
    sharpe = (series.mean() * trading_days - risk_free_rate) / ann_vol
    print(f"\n===== {label} =====")
    print(f"Total Return:      {total_ret:.2%}")
    print(f"Annual Volatility: {ann_vol:.2%}")
    print(f"Sharpe Ratio:      {sharpe:.2f}")

compute_metrics(rolling_max_series, "ROLLING MAX SHARPE (Vol-Capped)")
compute_metrics(rolling_min_series, "ROLLING MIN VOLATILITY")

print("\n=== ENGINE DEPLOYED FOR FGKFX TICKERS ===")

# ==========================================================
# DUAL-STRATEGY PROFESSIONAL PORTFOLIO ENGINE
# IBM vs GOOGL vs IONQ
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
# CONFIGURATION
# ----------------------------------------------------------
tickers = ["IBM", "GOOGL", "IONQ"]
start_date = "2018-01-01"
risk_free_rate = 0.02
trading_days = 252
rebalance_frequency = 63  # approx quarterly
rolling_window = 252  # 1-year rolling window

# Volatility caps (max allocation per asset in rolling Max Sharpe)
vol_caps = {"IBM": 0.5, "GOOGL": 0.7, "IONQ": 0.15}

# ----------------------------------------------------------
# DOWNLOAD DATA
# ----------------------------------------------------------
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

def portfolio_performance(weights, mean_ret=mean_returns, cov=cov_matrix):
    ret = np.dot(weights, mean_ret)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    sharpe = (ret - risk_free_rate) / vol
    return ret, vol, sharpe

# Max Sharpe
def negative_sharpe(weights):
    return -portfolio_performance(weights)[2]

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0,1) for _ in range(len(tickers)))
initial_weights = np.array(len(tickers) * [1./len(tickers)])

opt_sharpe = minimize(negative_sharpe, initial_weights, method='SLSQP',
                      bounds=bounds, constraints=constraints)
max_sharpe_weights = opt_sharpe.x
max_sharpe_perf = portfolio_performance(max_sharpe_weights)

# Min Volatility
def portfolio_volatility(weights, mean_ret=mean_returns, cov=cov_matrix):
    return portfolio_performance(weights, mean_ret, cov)[1]

opt_vol = minimize(portfolio_volatility, initial_weights, method='SLSQP',
                   bounds=bounds, constraints=constraints)
min_vol_weights = opt_vol.x
min_vol_perf = portfolio_performance(min_vol_weights)

# ----------------------------------------------------------
# EFFICIENT FRONTIER
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
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', alpha=0.7)
plt.colorbar(label="Sharpe Ratio")
plt.scatter(max_sharpe_perf[1], max_sharpe_perf[0], color='red', s=200, label="Max Sharpe")
plt.scatter(min_vol_perf[1], min_vol_perf[0], color='blue', s=200, label="Min Volatility")
plt.xlabel("Volatility")
plt.ylabel("Return")
plt.title("Efficient Frontier")
plt.legend()
plt.show()

# ----------------------------------------------------------
# PRINT STATIC PORTFOLIOS
# ----------------------------------------------------------
print("\n===== STATIC MAX SHARPE PORTFOLIO =====")
for t, w in zip(tickers, max_sharpe_weights):
    print(f"{t}: {w:.2%}")
print(f"Return: {max_sharpe_perf[0]:.2%}")
print(f"Volatility: {max_sharpe_perf[1]:.2%}")
print(f"Sharpe: {max_sharpe_perf[2]:.2f}")

print("\n===== STATIC MIN VOLATILITY PORTFOLIO =====")
for t, w in zip(tickers, min_vol_weights):
    print(f"{t}: {w:.2%}")
print(f"Return: {min_vol_perf[0]:.2%}")
print(f"Volatility: {min_vol_perf[1]:.2%}")
print(f"Sharpe: {min_vol_perf[2]:.2f}")

# ----------------------------------------------------------
# ROLLING STRATEGY BACKTEST
# ----------------------------------------------------------
rolling_max_sharpe_values = []
rolling_min_vol_values = []
dates = []

for i in range(rolling_window + rebalance_frequency, len(returns), rebalance_frequency):
    window_returns = returns.iloc[i-rolling_window:i]
    if window_returns.shape[0] < rolling_window:
        continue

    mean_r = window_returns.mean() * trading_days
    cov_m = window_returns.cov() * trading_days

    # --- Volatility-capped Max Sharpe
    def neg_sharpe_cap(weights):
        return -portfolio_performance(weights, mean_r, cov_m)[2]

    # bounds with volatility caps
    bounds_cap = tuple((0, vol_caps[t]) for t in tickers)
    opt_cap = minimize(neg_sharpe_cap, initial_weights, method='SLSQP',
                       bounds=bounds_cap, constraints=constraints)
    w_cap = opt_cap.x

    # --- Rolling Min Volatility
    opt_min_vol = minimize(portfolio_volatility, initial_weights, method='SLSQP',
                           bounds=bounds, constraints=constraints, args=(mean_r, cov_m))
    w_min_vol = opt_min_vol.x

    period_returns = returns.iloc[i:i+rebalance_frequency]
    rolling_max_sharpe_values.extend(period_returns @ w_cap)
    rolling_min_vol_values.extend(period_returns @ w_min_vol)
    dates.extend(period_returns.index)

rolling_max_series = pd.Series(rolling_max_sharpe_values, index=dates)
rolling_min_series = pd.Series(rolling_min_vol_values, index=dates)

# ----------------------------------------------------------
# PLOT ROLLING STRATEGIES
# ----------------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot((1 + rolling_max_series).cumprod(), label="Rolling Max Sharpe (Vol-Capped)", color="red")
plt.plot((1 + rolling_min_series).cumprod(), label="Rolling Min Volatility", color="blue")
plt.title("Dual Strategy Cumulative Returns (Quarterly Rebalance)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------------------
# ROLLING STRATEGY METRICS
# ----------------------------------------------------------
def compute_metrics(series):
    total_ret = (1 + series).prod() - 1
    ann_vol = series.std() * np.sqrt(trading_days)
    sharpe = (series.mean() * trading_days - risk_free_rate) / ann_vol
    return total_ret, ann_vol, sharpe

max_ret, max_vol, max_sharpe_roll = compute_metrics(rolling_max_series)
min_ret, min_vol, min_sharpe_roll = compute_metrics(rolling_min_series)

print("\n===== ROLLING MAX SHARPE (Vol-Capped) =====")
print(f"Total Return: {max_ret:.2%}")
print(f"Annual Volatility: {max_vol:.2%}")
print(f"Sharpe Ratio: {max_sharpe_roll:.2f}")

print("\n===== ROLLING MIN VOLATILITY =====")
print(f"Total Return: {min_ret:.2%}")
print(f"Annual Volatility: {min_vol:.2%}")
print(f"Sharpe Ratio: {min_sharpe_roll:.2f}")

print("\n=== DUAL-STRATEGY PROFESSIONAL PORTFOLIO ENGINE DEPLOYED SUCCESSFULLY ===")

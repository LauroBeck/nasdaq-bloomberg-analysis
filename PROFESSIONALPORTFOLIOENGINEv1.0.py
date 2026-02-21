# ==========================================================
# PROFESSIONAL PORTFOLIO ENGINE v1.0
# IBM vs GOOGL vs IONQ
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

def portfolio_performance(weights):
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (ret - risk_free_rate) / vol
    return ret, vol, sharpe

def negative_sharpe(weights):
    return -portfolio_performance(weights)[2]

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0,1) for _ in range(len(tickers)))
initial_weights = np.array(len(tickers) * [1./len(tickers)])

# Max Sharpe Portfolio
opt_sharpe = minimize(negative_sharpe, initial_weights,
                      method='SLSQP', bounds=bounds,
                      constraints=constraints)
max_sharpe_weights = opt_sharpe.x
max_sharpe_perf = portfolio_performance(max_sharpe_weights)

# Min Volatility Portfolio
def portfolio_volatility(weights):
    return portfolio_performance(weights)[1]

opt_vol = minimize(portfolio_volatility, initial_weights,
                   method='SLSQP', bounds=bounds,
                   constraints=constraints)
min_vol_weights = opt_vol.x
min_vol_perf = portfolio_performance(min_vol_weights)

# ----------------------------------------------------------
# EFFICIENT FRONTIER SIMULATION
# ----------------------------------------------------------
num_portfolios = 5000
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    ret, vol, sharpe = portfolio_performance(weights)
    results[0,i] = ret
    results[1,i] = vol
    results[2,i] = sharpe

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
print("\n===== MAX SHARPE PORTFOLIO =====")
for ticker, weight in zip(tickers, max_sharpe_weights):
    print(f"{ticker}: {weight:.2%}")
print(f"Return: {max_sharpe_perf[0]:.2%}")
print(f"Volatility: {max_sharpe_perf[1]:.2%}")
print(f"Sharpe: {max_sharpe_perf[2]:.2f}")

print("\n===== MIN VOLATILITY PORTFOLIO =====")
for ticker, weight in zip(tickers, min_vol_weights):
    print(f"{ticker}: {weight:.2%}")
print(f"Return: {min_vol_perf[0]:.2%}")
print(f"Volatility: {min_vol_perf[1]:.2%}")
print(f"Sharpe: {min_vol_perf[2]:.2f}")

# ----------------------------------------------------------
# ROLLING MAX SHARPE BACKTEST (Quarterly Rebalance)
# ----------------------------------------------------------
portfolio_values = []
dates = []

for i in range(252 + rebalance_frequency, len(returns), rebalance_frequency):
    window_returns = returns.iloc[i-252:i]
    if window_returns.shape[0] < 252:
        continue  # skip incomplete window

    mean_r = window_returns.mean() * trading_days
    cov_m = window_returns.cov() * trading_days

    def portfolio_perf(w):
        r = np.dot(w, mean_r)
        v = np.sqrt(np.dot(w.T, np.dot(cov_m, w)))
        s = (r - risk_free_rate) / v
        return -s

    opt = minimize(portfolio_perf, initial_weights,
                   method='SLSQP', bounds=bounds,
                   constraints=constraints)
    weights = opt.x

    period_returns = returns.iloc[i:i+rebalance_frequency]
    portfolio_period = period_returns @ weights
    portfolio_values.extend(portfolio_period)
    dates.extend(period_returns.index)

portfolio_series = pd.Series(portfolio_values, index=dates)

# ----------------------------------------------------------
# PLOT ROLLING MAX SHARPE STRATEGY
# ----------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot((1 + portfolio_series).cumprod())
plt.title("Rolling Max Sharpe Strategy (Quarterly Rebalance)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.show()

# ----------------------------------------------------------
# BACKTEST METRICS
# ----------------------------------------------------------
total_return = (1 + portfolio_series).prod() - 1
annual_vol = portfolio_series.std() * np.sqrt(trading_days)
sharpe_ratio = (portfolio_series.mean() * trading_days - risk_free_rate) / annual_vol

print("\n===== ROLLING MAX SHARPE STRATEGY METRICS =====")
print(f"Total Return: {total_return:.2%}")
print(f"Annual Volatility: {annual_vol:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

print("\n=== PROFESSIONAL PORTFOLIO ENGINE DEPLOYED SUCCESSFULLY ===")

# ==========================================================
# PROFESSIONAL PORTFOLIO OPTIMIZER
# IBM vs GOOGL vs IONQ
# ==========================================================

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.style.use("dark_background")

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
tickers = ["IBM", "GOOGL", "IONQ"]
start_date = "2018-01-01"
risk_free_rate = 0.02
trading_days = 252

# ----------------------------------------------------------
# DOWNLOAD DATA
# ----------------------------------------------------------
raw = yf.download(tickers, start=start_date, auto_adjust=True)

if isinstance(raw.columns, pd.MultiIndex):
    prices = raw["Close"]
else:
    prices = raw[["Close"]]

returns = prices.pct_change(fill_method=None).dropna()


mean_returns = returns.mean() * trading_days
cov_matrix = returns.cov() * trading_days

# ----------------------------------------------------------
# PORTFOLIO FUNCTIONS
# ----------------------------------------------------------
def portfolio_performance(weights):
    returns_p = np.dot(weights, mean_returns)
    volatility_p = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_p = (returns_p - risk_free_rate) / volatility_p
    return returns_p, volatility_p, sharpe_p

def negative_sharpe(weights):
    return -portfolio_performance(weights)[2]

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0,1) for _ in range(len(tickers)))
initial_weights = np.array(len(tickers) * [1./len(tickers)])

# ----------------------------------------------------------
# MAX SHARPE PORTFOLIO
# ----------------------------------------------------------
opt_sharpe = minimize(
    negative_sharpe,
    initial_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

max_sharpe_weights = opt_sharpe.x
max_sharpe_perf = portfolio_performance(max_sharpe_weights)

# ----------------------------------------------------------
# MIN VOLATILITY PORTFOLIO
# ----------------------------------------------------------
def portfolio_volatility(weights):
    return portfolio_performance(weights)[1]

opt_vol = minimize(
    portfolio_volatility,
    initial_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

min_vol_weights = opt_vol.x
min_vol_perf = portfolio_performance(min_vol_weights)
# ----------------------------------------------------------
# ROLLING MAX SHARPE BACKTEST (Quarterly Rebalance)
# ----------------------------------------------------------

from scipy.optimize import minimize

rebalance_frequency = 63  # approx quarterly
portfolio_values = []
dates = []

for i in range(rebalance_frequency, len(returns), rebalance_frequency):
    # Use 1-year rolling window for mean/cov calculation
    window_returns = returns.iloc[i-252:i]  # last 252 trading days

    mean_r = window_returns.mean() * trading_days
    cov_m = window_returns.cov() * trading_days

    def portfolio_perf(w):
        r = np.dot(w, mean_r)
        v = np.sqrt(np.dot(w.T, np.dot(cov_m, w)))
        s = (r - risk_free_rate) / v
        return -s

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for _ in range(len(tickers)))
    initial_weights = np.array(len(tickers) * [1./len(tickers)])

    opt = minimize(portfolio_perf,
                   initial_weights,
                   method='SLSQP',
                   bounds=bounds,
                   constraints=constraints)

    weights = opt.x

    period_returns = returns.iloc[i:i+rebalance_frequency]
    portfolio_period = period_returns @ weights #portfolio daily return
    portfolio_values.extend(portfolio_period)
    dates.extend(period_returns.index)

portfolio_series = pd.Series(portfolio_values, index=dates)

# Plot the rolling strategy
plt.figure(figsize=(10,6))
plt.plot((1 + portfolio_series).cumprod())
plt.title("Rolling Max Sharpe Strategy (Quarterly Rebalance)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.show()

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

# ----------------------------------------------------------
# PLOT EFFICIENT FRONTIER
# ----------------------------------------------------------
plt.figure(figsize=(10,6))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis')
plt.colorbar(label="Sharpe Ratio")

plt.scatter(max_sharpe_perf[1], max_sharpe_perf[0],
            color='red', s=200, label="Max Sharpe")

plt.scatter(min_vol_perf[1], min_vol_perf[0],
            color='blue', s=200, label="Min Volatility")

plt.xlabel("Volatility")
plt.ylabel("Return")
plt.title("Efficient Frontier")
plt.legend()
plt.show()

# ----------------------------------------------------------
# PRINT RESULTS
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

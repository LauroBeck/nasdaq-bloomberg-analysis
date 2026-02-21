# ==========================================================
# QUANTUM EQUITY ADVANCED ANALYTICS TERMINAL
# IBM vs GOOGL vs IONQ
# ==========================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm

plt.style.use("dark_background")

# ----------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------
tickers = ["IBM", "GOOGL", "IONQ"]
start_date = "2018-01-01"
risk_free_rate = 0.02
trading_days = 252

# ----------------------------------------------------------
# 2. DOWNLOAD DATA
# ----------------------------------------------------------
data = yf.download(tickers, start=start_date, auto_adjust=True)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

prices = data[['Close']]
returns = prices.pct_change().dropna()

# ----------------------------------------------------------
# 3. PERFORMANCE METRICS
# ----------------------------------------------------------
annual_returns = returns.mean() * trading_days
annual_vol = returns.std() * np.sqrt(trading_days)

sharpe = (annual_returns - risk_free_rate) / annual_vol

downside = returns.copy()
downside[downside > 0] = 0
sortino = (annual_returns - risk_free_rate) / (downside.std() * np.sqrt(trading_days))

rolling_max = prices.cummax()
drawdown = prices / rolling_max - 1
max_dd = drawdown.min()

metrics = pd.DataFrame({
    "Annual Return": annual_returns,
    "Volatility": annual_vol,
    "Sharpe": sharpe,
    "Sortino": sortino,
    "Max Drawdown": max_dd
})

print("\n===== PERFORMANCE METRICS =====")
print(metrics)

# ----------------------------------------------------------
# 4. CUMULATIVE RETURNS PLOT
# ----------------------------------------------------------
cum_returns = (1 + returns).cumprod()

plt.figure(figsize=(12,6))
for col in cum_returns.columns:
    plt.plot(cum_returns[col], label=col)

plt.title("Cumulative Returns")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------------------
# 5. MONTE CARLO SIMULATION (IBM)
# ----------------------------------------------------------
S0 = prices["Close"].iloc[-1]
mu = annual_returns["Close"]
sigma = annual_vol["Close"]
T = 1
simulations = 1000

dt = 1/trading_days
price_paths = np.zeros((trading_days, simulations))

for i in range(simulations):
    price = S0
    for t in range(trading_days):
        price *= np.exp((mu - 0.5*sigma**2)*dt +
                        sigma*np.sqrt(dt)*np.random.normal())
        price_paths[t, i] = price

plt.figure(figsize=(12,6))
plt.plot(price_paths[:, :50])
plt.title("Monte Carlo Simulation - IBM")
plt.show()

# ----------------------------------------------------------
# 6. ARIMA FORECAST (IBM)
# ----------------------------------------------------------
ibm_series = prices["Close"]

model = ARIMA(ibm_series, order=(2,1,2))
model_fit = model.fit()

forecast = model_fit.forecast(steps=60)

plt.figure(figsize=(10,5))
plt.plot(ibm_series[-200:], label="Historical")
plt.plot(forecast, label="ARIMA Forecast", color="red")
plt.title("IBM ARIMA Forecast (60 Days)")
plt.legend()
plt.show()


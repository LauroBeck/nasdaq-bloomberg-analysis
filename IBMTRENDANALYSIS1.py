# ==========================================================
# IBM QUANTUM + STOCK TREND ANALYSIS (UPDATED VERSION)
# Compatible with Python 3.13 + pandas 2.x + yfinance latest
# ==========================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# -----------------------------
# 1. DOWNLOAD IBM STOCK DATA
# -----------------------------
ticker = "IBM"
start_date = "2015-01-01"

data = yf.download(ticker, start=start_date, auto_adjust=True)

# Handle MultiIndex columns safely
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Use adjusted Close (auto_adjust=True already adjusts)
data = data[['Close']].rename(columns={'Close': 'Price'})

# -----------------------------
# 2. MOVING AVERAGES
# -----------------------------
data['MA50'] = data['Price'].rolling(50).mean()
data['MA200'] = data['Price'].rolling(200).mean()

# -----------------------------
# 3. RETURNS + RISK METRICS
# -----------------------------
data['Daily Return'] = data['Price'].pct_change()

trading_days = 252
years = len(data) / trading_days

annual_return = (data['Price'].iloc[-1] / data['Price'].iloc[0])**(1/years) - 1
annual_volatility = data['Daily Return'].std() * np.sqrt(trading_days)

risk_free_rate = 0.02
sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

# Max Drawdown
rolling_max = data['Price'].cummax()
drawdown = data['Price'] / rolling_max - 1
max_drawdown = drawdown.min()

# -----------------------------
# 4. RSI CALCULATION
# -----------------------------
window = 14
delta = data['Price'].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(window).mean()
avg_loss = loss.rolling(window).mean()

rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# -----------------------------
# 5. PRICE + MOVING AVERAGE PLOT
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(data['Price'], label='IBM Price')
plt.plot(data['MA50'], label='50-Day MA')
plt.plot(data['MA200'], label='200-Day MA')
plt.title("IBM Stock Trend Analysis")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 6. RSI PLOT
# -----------------------------
plt.figure(figsize=(10,4))
plt.plot(data['RSI'], label='RSI')
plt.axhline(70, linestyle='--')
plt.axhline(30, linestyle='--')
plt.title("IBM RSI Indicator")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 7. PRINT STOCK METRICS
# -----------------------------
print("====== IBM STOCK METRICS ======")
print(f"Annual Return: {annual_return:.2%}")
print(f"Annual Volatility: {annual_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# ==========================================================
# 8. IBM QUANTUM QUBIT GROWTH TREND (Manual Dataset)
# ==========================================================

qubit_data = {
    "Year": [2017, 2019, 2021, 2023, 2025],
    "Qubits": [5, 20, 127, 433, 1121]
}

df_q = pd.DataFrame(qubit_data)

# Linear regression model
slope, intercept, r_value, p_value, std_err = linregress(
    df_q["Year"], df_q["Qubits"]
)

future_years = np.arange(2017, 2031)
forecast = intercept + slope * future_years

# Plot Qubit Growth
plt.figure(figsize=(10,6))
plt.scatter(df_q["Year"], df_q["Qubits"], label="Actual Qubits")
plt.plot(future_years, forecast, label="Trend Projection")
plt.title("IBM Quantum Qubit Growth Trend")
plt.xlabel("Year")
plt.ylabel("Number of Qubits")
plt.legend()
plt.grid(True)
plt.show()

print("\n====== IBM QUBIT GROWTH MODEL ======")
print(f"Qubits Added Per Year (Trend Slope): {slope:.2f}")
print(f"Model R-Squared: {r_value**2:.4f}")

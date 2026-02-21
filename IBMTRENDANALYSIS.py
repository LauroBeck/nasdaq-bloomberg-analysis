# ==========================================================
# IBM QUANTUM + STOCK TREND ANALYSIS (ALL-IN-ONE SCRIPT)
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

data = yf.download(ticker, start=start_date)
data = data[['Adj Close']]
data.rename(columns={'Adj Close': 'Price'}, inplace=True)

# -----------------------------
# 2. CALCULATE MOVING AVERAGES
# -----------------------------
data['MA50'] = data['Price'].rolling(50).mean()
data['MA200'] = data['Price'].rolling(200).mean()

# -----------------------------
# 3. CALCULATE RETURNS
# -----------------------------
data['Daily Return'] = data['Price'].pct_change()
annual_return = (data['Price'].iloc[-1] / data['Price'].iloc[0]) ** (1/((len(data))/252)) - 1
annual_volatility = data['Daily Return'].std() * np.sqrt(252)

# -----------------------------
# 4. RSI CALCULATION
# -----------------------------
window = 14
delta = data['Price'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window).mean()
avg_loss = loss.rolling(window).mean()

rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# -----------------------------
# 5. PLOT STOCK + MA
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
# 6. PLOT RSI
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
# 7. PRINT METRICS
# -----------------------------
print("====== IBM STOCK METRICS ======")
print(f"Annual Return: {annual_return:.2%}")
print(f"Annual Volatility: {annual_volatility:.2%}")

# ==========================================================
# 8. IBM QUANTUM QUBIT GROWTH TREND
# (Manual Dataset - Update as needed)
# ==========================================================

qubit_data = {
    "Year": [2017, 2019, 2021, 2023, 2025],
    "Qubits": [5, 20, 127, 433, 1121]
}

df_q = pd.DataFrame(qubit_data)

# Linear regression forecast
slope, intercept, r_value, p_value, std_err = linregress(df_q["Year"], df_q["Qubits"])

future_years = np.arange(2017, 2031)
forecast = intercept + slope * future_years

# Plot Qubit Trend
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
print(f"Trend slope (qubits per year): {slope:.2f}")
print(f"R-squared: {r_value**2:.4f}")

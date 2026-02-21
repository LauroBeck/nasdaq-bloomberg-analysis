import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ----------------------------
# 1. Download market data
# ----------------------------

wti = yf.download("CL=F", start="2020-01-01")['Close']
brent = yf.download("BZ=F", start="2020-01-01")['Close']
dxy = yf.download("DX-Y.NYB", start="2020-01-01")['Close']

df = pd.DataFrame({
    'WTI': wti,
    'Brent': brent,
    'DXY': dxy
}).dropna()

# ----------------------------
# 2. Transform to returns
# ----------------------------

returns = np.log(df / df.shift(1))
df['WTI_ret'] = returns['WTI']
df['Brent_ret'] = returns['Brent']
df['DXY_ret'] = returns['DXY']

# ----------------------------
# 3. Feature Engineering
# ----------------------------

df['volatility'] = df['WTI_ret'].rolling(10).std()
df['momentum'] = df['WTI_ret'].rolling(5).mean()

df['lag1'] = df['WTI_ret'].shift(1)
df['lag2'] = df['WTI_ret'].shift(2)
df['lag3'] = df['WTI_ret'].shift(3)

# Target = next-day return
df['target'] = df['WTI_ret'].shift(-1)

df.dropna(inplace=True)

features = [
    'lag1', 'lag2', 'lag3',
    'volatility', 'momentum',
    'Brent_ret', 'DXY_ret'
]

X = df[features]
y = df['target']

# ----------------------------
# 4. Time-series split
# ----------------------------

split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ----------------------------
# 5. Model (nonlinear)
# ----------------------------

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

df['predicted_ret'] = model.predict(X)

# ----------------------------
# 6. Convert return → price
# ----------------------------

df['predicted_price'] = df['WTI'].shift(1) * np.exp(df['predicted_ret'])

# ----------------------------
# 7. Plot
# ----------------------------

plt.figure()
plt.plot(df.index, df['WTI'], label="Actual WTI")
plt.plot(df.index, df['predicted_price'], label="Predicted WTI")
plt.legend()
plt.title("WTI NYMEX Prediction Model (RF + Market Drivers)")
plt.show()

# ----------------------------
# 8. Next-day forecast
# ----------------------------

last_row = X.iloc[-1].values.reshape(1, -1)
next_ret = model.predict(last_row)[0]

last_price = df['WTI'].iloc[-1]
next_price = last_price * np.exp(next_ret)

print(f"Last WTI price: {last_price:.2f}")
print(f"Predicted next WTI price: {next_price:.2f}")

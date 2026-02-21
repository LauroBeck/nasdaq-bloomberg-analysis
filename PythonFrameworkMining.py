import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# -----------------------------------
# CONFIGURATION
# -----------------------------------

stocks = {
    "Rio Tinto": "RIO",
    "BHP Group": "BHP",
    "BlueScope Steel": "BSL.AX",
    "Pilbara Minerals": "PLS.AX"
}

market_index = "^AXJO"     # ASX 200 proxy
commodity_proxy = "PICK"   # Global miners ETF proxy

projection_year = 2026
history_period = "5y"

# -----------------------------------
# DATA DOWNLOAD
# -----------------------------------

tickers = list(stocks.values()) + [market_index, commodity_proxy]

data = yf.download(tickers, period=history_period)

prices = data["Close"]
dividends = data["Dividends"]

# -----------------------------------
# DIVIDEND PROJECTION ENGINE
# -----------------------------------

# -----------------------------------
# DIVIDEND PROJECTION ENGINE (FIXED)
# -----------------------------------

div_proj = {}

for name, ticker in stocks.items():

    tk = yf.Ticker(ticker)
    div_series = tk.dividends   # ✅ Always correct source

    if div_series is None or len(div_series) == 0:
        div_proj[name] = np.nan
        continue

    div_series = div_series[div_series > 0]

    if len(div_series) < 2:
        div_proj[name] = np.nan
        continue

    annual_div = div_series.resample("Y").sum()
    growth = annual_div.pct_change().mean()

    last_div = annual_div.iloc[-1]
    years_forward = projection_year - datetime.now().year

    projected_div = last_div * (1 + growth) ** years_forward

    div_proj[name] = float(projected_div)


# -----------------------------------
# FACTOR MODEL
# -----------------------------------

returns = prices.pct_change().dropna()

factors = {}

for name, ticker in stocks.items():

    stock_ret = returns[ticker]
    market_ret = returns[market_index]
    commodity_ret = returns[commodity_proxy]

    # Beta vs Market
    beta = np.cov(stock_ret, market_ret)[0, 1] / np.var(market_ret)

    # Momentum (6M)
    momentum = prices[ticker].pct_change(126).iloc[-1]

    # Volatility
    vol = stock_ret.std() * np.sqrt(252)

    # Commodity Sensitivity
    commodity_beta = np.cov(stock_ret, commodity_ret)[0, 1] / np.var(commodity_ret)

    # Relative Strength
    rel_strength = prices[ticker].iloc[-1] / prices[commodity_proxy].iloc[-1]

    factors[name] = {
        "beta": beta,
        "momentum": momentum,
        "volatility": vol,
        "commodity_beta": commodity_beta,
        "relative_strength": rel_strength
    }

factor_df = pd.DataFrame(factors).T

# -----------------------------------
# BLOOMBERG-STYLE DECISION ENGINE
# -----------------------------------

decisions = {}

for stock in factor_df.index:

    beta = factor_df.loc[stock, "beta"]
    momentum = factor_df.loc[stock, "momentum"]
    vol = factor_df.loc[stock, "volatility"]
    commodity_beta = factor_df.loc[stock, "commodity_beta"]

    decision = "HOLD"

    if momentum > 0.15 and commodity_beta > 0.5:
        decision = "STRONG BUY"

    elif momentum > 0.05 and beta > 0.8:
        decision = "BUY"

    elif momentum < -0.10 and vol > factor_df["volatility"].mean():
        decision = "SELL"

    elif momentum < -0.20:
        decision = "STRONG SELL"

    decisions[stock] = decision

# -----------------------------------
# OUTPUT
# -----------------------------------

print("\n=== DIVIDEND PROJECTION 2026 ===")
for k, v in div_proj.items():
    print(f"{k}: {v:.2f} (model estimate)")

print("\n=== FACTOR MODEL SNAPSHOT ===")
print(factor_df.round(3))

print("\n=== DECISION ENGINE ===")
for k, v in decisions.items():
    print(f"{k}: {v}")

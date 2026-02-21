import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide")

# -----------------------------------
# CONFIGURATION
# -----------------------------------

stocks = {
    "Rio Tinto": "RIO",
    "BHP Group": "BHP",
    "Pilbara Minerals": "PLS.AX"
}

commodity_proxies = {
    "Global Miners ETF": "PICK",
    "Copper Miners ETF": "COPX",
    "Lithium ETF": "LIT"
}

market_index = "^GSPC"
history_period = "2y"
projection_year = 2026

# -----------------------------------
# DOWNLOAD PRICE DATA
# -----------------------------------

tickers = list(stocks.values()) + list(commodity_proxies.values()) + [market_index]
data = yf.download(tickers, period=history_period)

prices = data["Close"]
returns = prices.pct_change().dropna()

# -----------------------------------
# DIVIDEND YIELD PROJECTION
# -----------------------------------

dividend_yields = {}

for name, ticker in stocks.items():

    tk = yf.Ticker(ticker)
    divs = tk.dividends

    if divs is None or divs.empty:
        dividend_yields[name] = np.nan
        continue

    annual_div = divs.resample("Y").sum()
    growth = annual_div.pct_change().mean()

    if np.isnan(growth):
        growth = 0

    last_div = annual_div.iloc[-1]
    years_forward = projection_year - datetime.now().year

    projected_div = last_div * (1 + growth) ** years_forward
    current_price = prices[ticker].iloc[-1]

    projected_yield = projected_div / current_price
    dividend_yields[name] = float(projected_yield)

# -----------------------------------
# FACTOR + CASH FLOW PROXY MODEL
# -----------------------------------

model_data = {}

for name, ticker in stocks.items():

    stock_ret = returns[ticker]
    miner_ret = returns[commodity_proxies["Global Miners ETF"]]

    beta = np.cov(stock_ret, miner_ret)[0, 1] / np.var(miner_ret)
    momentum = prices[ticker].pct_change(126).iloc[-1]
    vol = stock_ret.std() * np.sqrt(252)

    # Cash Flow Proxy
    cash_flow_score = (
        0.6 * momentum +
        0.3 * beta -
        0.2 * vol
    )

    model_data[name] = {
        "price": prices[ticker].iloc[-1],
        "momentum": momentum,
        "beta": beta,
        "volatility": vol,
        "cash_flow_score": cash_flow_score,
        "projected_yield": dividend_yields[name]
    }

df = pd.DataFrame(model_data).T

# -----------------------------------
# COMMODITY REGIME DETECTION
# -----------------------------------

miner_momentum = prices[commodity_proxies["Global Miners ETF"]].pct_change(126).iloc[-1]

if miner_momentum > 0.10:
    commodity_regime = "RISK ON / COMMODITY STRENGTH"
elif miner_momentum < -0.10:
    commodity_regime = "RISK OFF / COMMODITY WEAKNESS"
else:
    commodity_regime = "NEUTRAL"

# -----------------------------------
# DECISION ENGINE
# -----------------------------------

decisions = {}

for stock in df.index:

    m = df.loc[stock, "momentum"]
    cf = df.loc[stock, "cash_flow_score"]
    yld = df.loc[stock, "projected_yield"]

    decision = "HOLD"

    if commodity_regime.startswith("RISK ON") and m > 0.10 and cf > 0:
        decision = "STRONG BUY"

    elif m > 0.05 and cf > 0:
        decision = "BUY"

    elif m < -0.10 and cf < 0:
        decision = "SELL"

    elif m < -0.20:
        decision = "STRONG SELL"

    if not np.isnan(yld) and yld > 0.05:
        decision += " (Yield Support)"

    decisions[stock] = decision

# -----------------------------------
# STREAMLIT UI
# -----------------------------------

st.title("Mining & Materials Quant Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Commodity Regime")
    st.write(commodity_regime)

with col2:
    st.subheader("Global Miners Momentum (6M)")
    st.write(f"{miner_momentum:.2%}")

st.divider()

st.subheader("Factor & Cash Flow Model")
st.dataframe(df.round(3), use_container_width=True)

st.divider()

st.subheader("Projected Dividend Yields (2026 Model)")
for k, v in dividend_yields.items():
    if np.isnan(v):
        st.write(f"{k}: No dividend data")
    else:
        st.write(f"{k}: {v:.2%}")

st.divider()

st.subheader("Decision Engine")
for k, v in decisions.items():
    st.write(f"{k}: {v}")

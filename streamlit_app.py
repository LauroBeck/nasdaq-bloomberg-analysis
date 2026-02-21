import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import streamlit as st

st.set_page_config(layout="wide")

# --------------------------------
# Configuration
# --------------------------------
assets = {
    "Brent": "BZ=F",
    "WTI": "CL=F",
    "Chevron": "CVX",
    "ExxonMobil": "XOM",
    "ConocoPhillips": "COP",
    "Shell": "SHEL",
    "BP": "BP",
    "Halliburton": "HAL",
    "Baker Hughes": "BKR",
    "Maersk": "AMKBY",
    "Hapag-Lloyd": "HPGLY"
}

start_date = "2023-01-01"

# --------------------------------
# Download Data
# --------------------------------
@st.cache_data
def load_data():
    data = yf.download(list(assets.values()), start=start_date, auto_adjust=True)
    close = data["Close"]
    close.columns = assets.keys()
    return close.dropna()

prices = load_data()
returns = prices.pct_change().dropna()

# --------------------------------
# Live Macro Variables
# --------------------------------
brent_price = float(prices["Brent"].iloc[-1])
wti_price = float(prices["WTI"].iloc[-1])

# --------------------------------
# Regression: Oil Betas
# --------------------------------
def compute_beta(stock_returns, oil_returns):
    X = sm.add_constant(oil_returns)
    model = sm.OLS(stock_returns, X).fit()
    return model.params[1]

oil_returns = returns["Brent"]

betas = {}

for col in returns.columns:
    if col not in ["Brent", "WTI"]:
        betas[col] = compute_beta(returns[col], oil_returns)

beta_df = pd.DataFrame.from_dict(betas, orient="index", columns=["Oil Beta"])

# --------------------------------
# Volatility Estimation
# --------------------------------
vol_df = returns.std() * np.sqrt(252)
vol_df = vol_df.to_frame("Annualized Volatility")

# --------------------------------
# Scenario Engine
# --------------------------------
st.sidebar.title("Scenario Shocks")

oil_shock = st.sidebar.slider("Oil Price Shock (%)", -30, 50, 0)
vol_shock = st.sidebar.slider("Volatility Shock (%)", 0, 100, 0)

shock_multiplier = 1 + oil_shock / 100
vol_multiplier = 1 + vol_shock / 100

scenario_prices = {}

for asset in betas.keys():
    beta = betas[asset]
    price = prices[asset].iloc[-1]

    shocked_price = price * (1 + beta * (oil_shock / 100))
    scenario_prices[asset] = shocked_price

scenario_df = pd.DataFrame.from_dict(
    scenario_prices, orient="index", columns=["Scenario Price"]
)

# --------------------------------
# UI – Dashboard
# --------------------------------
st.title("Macro-Oil Risk Dashboard")

col1, col2 = st.columns(2)

col1.metric("Brent (BZ=F)", round(brent_price, 2))
col2.metric("WTI (CL=F)", round(wti_price, 2))

st.subheader("Oil Sensitivity (Regression Betas)")
st.dataframe(beta_df.sort_values(by="Oil Beta", ascending=False))

st.subheader("Market Volatility")
st.dataframe((vol_df * vol_multiplier).sort_values(
    by="Annualized Volatility", ascending=False
))

st.subheader("Scenario Shock Impact")
st.dataframe(scenario_df.sort_values(by="Scenario Price", ascending=False))

# Optional chart
st.subheader("Brent Price History")
st.line_chart(prices["Brent"])

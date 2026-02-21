import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import streamlit as st

st.set_page_config(layout="wide")

# --------------------------------
# Asset Universe
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
# Robust Yahoo Loader
# --------------------------------
@st.cache_data
def load_prices():
    frames = []
    failed = []

    for name, ticker in assets.items():
        try:
            df = yf.download(
                ticker,
                start=start_date,
                auto_adjust=True,
                progress=False
            )

            if df is None or df.empty:
                failed.append(name)
                continue

            frames.append(df["Close"].rename(name))

        except Exception:
            failed.append(name)

    if not frames:
        st.error("All market data downloads failed.")
        st.stop()

    prices = pd.concat(frames, axis=1).dropna()

    if failed:
        st.warning(f"Skipped failed tickers: {', '.join(failed)}")

    return prices

prices = load_prices()
returns = prices.pct_change().dropna()

# --------------------------------
# Live Macro Variables
# --------------------------------
brent_price = float(prices["Brent"].iloc[-1])
wti_price = float(prices["WTI"].iloc[-1])

# --------------------------------
# Oil Beta Regression
# --------------------------------
def compute_beta(y, x):
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model.params[1]

oil_returns = returns["Brent"]

betas = {}
valid_assets = [c for c in returns.columns if c not in ["Brent", "WTI"]]

for asset in valid_assets:
    try:
        betas[asset] = compute_beta(returns[asset], oil_returns)
    except Exception:
        pass

beta_df = pd.DataFrame.from_dict(
    betas, orient="index", columns=["Oil Beta"]
)

# --------------------------------
# Volatility Engine
# --------------------------------
vol_df = returns.std() * np.sqrt(252)
vol_df = vol_df.to_frame("Annualized Volatility")

# --------------------------------
# Scenario Engine
# --------------------------------
st.sidebar.title("Scenario Engine")

oil_shock = st.sidebar.slider("Oil Shock (%)", -40, 60, 0)
vol_shock = st.sidebar.slider("Volatility Shock (%)", 0, 150, 0)

oil_move = oil_shock / 100
vol_multiplier = 1 + vol_shock / 100

scenario = {}

for asset, beta in betas.items():
    spot = prices[asset].iloc[-1]
    shocked = spot * (1 + beta * oil_move)
    scenario[asset] = shocked

scenario_df = pd.DataFrame.from_dict(
    scenario, orient="index", columns=["Scenario Price"]
)

# --------------------------------
# UI Layout
# --------------------------------
st.title("Macro Oil Risk Dashboard")

col1, col2 = st.columns(2)
col1.metric("Brent (BZ=F)", round(brent_price, 2))
col2.metric("WTI (CL=F)", round(wti_price, 2))

st.subheader("Oil Sensitivity Betas (Regression)")
st.dataframe(beta_df.sort_values(by="Oil Beta", ascending=False))

st.subheader("Market Volatility")
st.dataframe(
    (vol_df * vol_multiplier).sort_values(
        by="Annualized Volatility", ascending=False
    )
)

st.subheader("Scenario Shock Impact")
st.dataframe(
    scenario_df.sort_values(by="Scenario Price", ascending=False)
)

st.subheader("Brent Price History")
st.line_chart(prices["Brent"])

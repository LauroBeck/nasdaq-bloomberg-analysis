import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import requests

st.set_page_config(layout="wide")

# -----------------------------
# Configuration
# -----------------------------

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
}

start_date = "2023-01-01"


# -----------------------------
# Connectivity Check
# -----------------------------

def check_connectivity():
    try:
        requests.get("https://query1.finance.yahoo.com", timeout=3)
        return True
    except Exception:
        return False


# -----------------------------
# Data Loader (Cached)
# -----------------------------

@st.cache_data(ttl=3600)
def load_prices():

    if not check_connectivity():
        return pd.DataFrame()

    tickers = list(assets.values())

    try:
        df = yf.download(
            tickers,
            start=start_date,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    frames = []

    for name, ticker in assets.items():

        try:
            # If multiple tickers, yfinance creates MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                close = df[ticker]["Close"]
            else:
                close = df["Close"]

            frames.append(close.rename(name))

        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1).dropna()

    return prices


# -----------------------------
# Main App
# -----------------------------

prices = load_prices()

if prices.empty:
    st.error("Failed to load price data.")
    st.stop()

returns = prices.pct_change().dropna()

# Latest prices
brent_price = float(prices["Brent"].iloc[-1])
wti_price = float(prices["WTI"].iloc[-1])

# -----------------------------
# UI
# -----------------------------

st.title("Macro Oil Dashboard")

col1, col2 = st.columns(2)
col1.metric("Brent", round(brent_price, 2))
col2.metric("WTI", round(wti_price, 2))

st.subheader("Brent Price")
st.line_chart(prices["Brent"])

st.subheader("Oil Majors vs Brent")
st.line_chart(prices)

st.subheader("Correlation Matrix")
st.dataframe(returns.corr().round(2))

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import streamlit as st
import requests
import time

st.set_page_config(layout="wide")

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

def check_connectivity():
    try:
        requests.get("https://query1.finance.yahoo.com", timeout=3)
        return True
    except Exception:
        return False

def download_with_retry(ticker, retries=3):
    for _ in range(retries):
        try:
            df = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        time.sleep(1)
    return None

def load_prices():

    if not check_connectivity():
        st.error("No Yahoo Finance connectivity.")
        return pd.DataFrame()

    frames = []
    failed = []

    for name, ticker in assets.items():
        df = download_with_retry(ticker)

        if df is None or df.empty:
            failed.append(name)
            continue

        frames.append(df["Close"].rename(name))

    if not frames:
        st.error("All downloads failed.")
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1).dropna()

    if failed:
        st.warning(f"Skipped: {', '.join(failed)}")

    return prices

prices = load_prices()

if prices.empty:
    st.stop()

returns = prices.pct_change().dropna()

brent_price = float(prices["Brent"].iloc[-1])
wti_price = float(prices["WTI"].iloc[-1])

st.title("Macro Oil Dashboard")

col1, col2 = st.columns(2)
col1.metric("Brent", round(brent_price, 2))
col2.metric("WTI", round(wti_price, 2))

st.line_chart(prices["Brent"])

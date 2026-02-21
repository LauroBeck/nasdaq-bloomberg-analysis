import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide")

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

ENERGY_FUTURES = {
    "NYMEX WTI": "CL=F",
    "Brent": "BZ=F",
    "RBOB Gas": "RB=F",
    "Nat Gas": "NG=F",
    "Heating Oil": "HO=F",
}

OIL_MAJORS = {
    "XOM": "XOM",
    "CVX": "CVX",
    "COP": "COP",
    "SHEL": "SHEL",
    "BP": "BP",
}

INDEX = {"Dow": "^DJI"}

# ---------------------------------------------------
# STYLE
# ---------------------------------------------------

st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
.metric-container { background-color: #1c1f26; padding:10px; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# DATA
# ---------------------------------------------------

@st.cache_data(ttl=300)
def load_prices(tickers, period="5d", interval="5m"):
    df = yf.download(list(tickers.values()),
                     period=period,
                     interval=interval,
                     auto_adjust=True,
                     progress=False,
                     group_by="ticker")
    return df

futures_df = load_prices(ENERGY_FUTURES)
majors_df = load_prices(OIL_MAJORS, period="5d", interval="1d")
index_df = load_prices(INDEX)

# ---------------------------------------------------
# LAYOUT
# ---------------------------------------------------

left, right = st.columns([2, 1])

# ===================================================
# LEFT PANEL
# ===================================================
with left:

    st.title("BIG OIL TERMINAL")

    # -------------------------------
    # Dow Intraday Block
    # -------------------------------

    st.subheader("Dow Jones Industrial Average")

    if isinstance(index_df.columns, pd.MultiIndex):
        dow = index_df["^DJI"]["Close"]
    else:
        dow = index_df["Close"]

    st.line_chart(dow)

    last = dow.iloc[-1]
    prev = dow.iloc[-2]
    change = last - prev
    pct = change / prev * 100

    color = "green" if change > 0 else "red"

    st.markdown(
        f"<h3 style='color:{color}'>{last:,.0f}  {change:+.2f} ({pct:+.2f}%)</h3>",
        unsafe_allow_html=True
    )

    # -------------------------------
    # Oil Majors Movers
    # -------------------------------

    st.subheader("Most Active Oil Majors")

    for name, ticker in OIL_MAJORS.items():

        if isinstance(majors_df.columns, pd.MultiIndex):
            series = majors_df[ticker]["Close"]
        else:
            series = majors_df["Close"]

        last = series.iloc[-1]
        prev = series.iloc[-2]
        change = last - prev
        pct = change / prev * 100

        color = "green" if change > 0 else "red"

        st.markdown(
            f"{name} — <span style='color:{color}'>{last:.2f} "
            f"{change:+.2f} ({pct:+.2f}%)</span>",
            unsafe_allow_html=True
        )

# ===================================================
# RIGHT PANEL
# ===================================================
with right:

    st.subheader("Energy Futures")

    for name, ticker in ENERGY_FUTURES.items():

        try:
            if isinstance(futures_df.columns, pd.MultiIndex):
                series = futures_df[ticker]["Close"]
            else:
                series = futures_df["Close"]

            last = series.iloc[-1]
            prev = series.iloc[-2]
            change = last - prev
            pct = change / prev * 100

            color = "green" if change > 0 else "red"

            st.markdown(
                f"**{name}**  \n"
                f"<span style='color:{color}'>{last:.2f} "
                f"{change:+.2f} ({pct:+.2f}%)</span>",
                unsafe_allow_html=True
            )

            st.line_chart(series.tail(60), height=80)

        except:
            continue

st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

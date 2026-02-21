import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

# ---------------------------
# STYLE (Bloomberg Dark)
# ---------------------------
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
.block-container { padding-top: 1rem; }
.metric { font-size: 18px !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# DATA DOWNLOAD
# ---------------------------

tickers = {
    "B500": "^GSPC",
    "EURP600": "^STOXX50E",
    "EDM": "EEM",
    "GB": "EWU",
    "APAC": "AAXJ",
    "WORLD": "URTH",
}

@st.cache_data
def load_data():
    df = yf.download(list(tickers.values()), period="6mo", auto_adjust=True, progress=False)
    return df["Close"]

prices = load_data()
returns = prices.pct_change().dropna()

# ---------------------------
# LAYOUT
# ---------------------------

left, right = st.columns([2, 1])

# ===========================
# LEFT PANEL
# ===========================
with left:

    st.markdown("## 📺 Bloomberg Television")

    st.image("https://via.placeholder.com/900x350.png?text=Live+TV+Feed")

    st.markdown("### EUR/USD TESTS 50-DMA SUPPORT")

    st.markdown("""
**Asian Stocks Fall, Oil Climbs With Iran in Focus: Markets Wrap**

- Caution resurfaces as US moves on Iran  
- Geopolitical risk dampens equities  
- Markets watching diplomatic resolution  
    """)

# ===========================
# RIGHT PANEL
# ===========================
with right:

    st.markdown("## Bloomberg Indices")

    latest = prices.iloc[-1]
    prev = prices.iloc[-2]

    for name, ticker in tickers.items():

        last = latest[ticker]
        change = last - prev[ticker]
        pct = (change / prev[ticker]) * 100

        color = "green" if change > 0 else "red"

        st.markdown(f"""
        **{name}**  
        {last:.2f}  
        <span style="color:{color}">{change:.2f} ({pct:.2f}%)</span>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Mini Charts")

    for ticker in tickers.values():
        st.line_chart(prices[ticker])

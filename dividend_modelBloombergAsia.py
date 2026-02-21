import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ------------------------------------------
# CONFIGURATION (EXPLICIT TICKERS)
# ------------------------------------------

stocks = {
    "Rio Tinto (US ADR)": "RIO",
    "BHP Group (US ADR)": "BHP",
    "Pilbara Minerals": "PLS.AX"
}

projection_year = 2026
price_period = "1y"

# ------------------------------------------
# DOWNLOAD PRICES
# ------------------------------------------

tickers = list(stocks.values())

print("\nDownloading price data...")

price_data = yf.download(tickers, period=price_period)["Close"]

if isinstance(price_data, pd.Series):
    price_data = price_data.to_frame()

# ------------------------------------------
# DIVIDEND ENGINE
# ------------------------------------------

results = []

current_year = datetime.now().year
years_forward = max(1, projection_year - current_year)

print("\nProcessing dividends...\n")

for name, ticker in stocks.items():

    tk = yf.Ticker(ticker)
    divs = tk.dividends

    if divs is None or divs.empty:
        print(f"{name}: No dividend history")
        continue

    divs = divs[divs > 0]

    if len(divs) < 2:
        print(f"{name}: Insufficient dividend data")
        continue

    annual_div = divs.resample("YE").sum()

    growth = annual_div.pct_change().mean()
    if np.isnan(growth):
        growth = 0

    last_div = annual_div.iloc[-1]

    projected_div = last_div * (1 + growth) ** years_forward

    try:
        current_price = float(price_data[ticker].iloc[-1])
    except:
        print(f"{name}: Price unavailable")
        continue

    projected_yield = projected_div / current_price

    results.append({
        "Company": name,
        "Ticker": ticker,
        "Price": current_price,
        "Last Dividend": float(last_div),
        "Growth Rate": float(growth),
        "Projected Dividend": float(projected_div),
        "Projected Yield": float(projected_yield)
    })

# ------------------------------------------
# OUTPUT
# ------------------------------------------

if not results:
    print("No dividend results available.")
    exit()

df = pd.DataFrame(results)
df = df.sort_values("Projected Yield", ascending=False)

print("\n================ DIVIDEND PROJECTION MODEL ================\n")

for _, row in df.iterrows():

    print(f"{row['Company']} ({row['Ticker']})")
    print(f"  Price: {row['Price']:.2f}")
    print(f"  Last Annual Dividend: {row['Last Dividend']:.2f}")
    print(f"  Avg Growth Rate: {row['Growth Rate']:.2%}")
    print(f"  Projected Dividend {projection_year}: {row['Projected Dividend']:.2f}")
    print(f"  Projected Yield {projection_year}: {row['Projected Yield']:.2%}")
    print("")

print("===========================================================\n")

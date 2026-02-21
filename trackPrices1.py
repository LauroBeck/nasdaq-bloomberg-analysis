import yfinance as yf
import pandas as pd
import numpy as np

# --------------------------------
# Configuration
# --------------------------------
tickers = {
    "Chevron": "CVX",
    "ConocoPhillips": "COP",
    "ExxonMobil": "XOM",
    "Shell": "SHEL",
    "BP": "BP",
    "Halliburton": "HAL",
    "Baker Hughes": "BKR",
    "Maersk": "AMKBY",        # ADR
    "Hapag-Lloyd": "HPGLY"    # ADR
}

start_date = "2024-01-01"
end_date = "2025-12-31"

# --------------------------------
# Download Market Data
# --------------------------------
data = yf.download(
    list(tickers.values()),
    start=start_date,
    end=end_date,
    group_by="ticker",
    auto_adjust=True,
    progress=True
)

# --------------------------------
# Build Close Price Table
# --------------------------------
prices = pd.DataFrame()

for company, ticker in tickers.items():
    prices[company] = data[ticker]["Close"]

latest_prices = prices.iloc[-1]

print("\nLatest Prices\n")
print(prices.tail())

# --------------------------------
# Derive Simple Oil Proxy (Heuristic)
# (You can replace with real Brent ticker = 'BZ=F')
# --------------------------------
oil_proxy = (
    latest_prices["ExxonMobil"] / prices["ExxonMobil"].mean()
) * 70

brent_price = round(oil_proxy, 2)

# Asia trade proxy from logistics strength
asia_trade_index = round(
    (latest_prices["Maersk"] + latest_prices["Hapag-Lloyd"]) /
    (prices["Maersk"].mean() + prices["Hapag-Lloyd"].mean()),
    2
)

volatility = 0.06

print("\nDerived Macro Inputs")
print("Brent proxy:", brent_price)
print("Asia Trade Index:", asia_trade_index)

# --------------------------------
# Earnings Sensitivity Model
# --------------------------------
companies = {
    "Chevron":        {"base": 18, "oil_beta": 0.45},
    "ConocoPhillips": {"base": 12, "oil_beta": 0.55},
    "ExxonMobil":     {"base": 24, "oil_beta": 0.40},
    "Shell":          {"base": 20, "oil_beta": 0.42},
    "BP":             {"base": 10, "oil_beta": 0.50},
    "Halliburton":    {"base": 4,  "oil_beta": 0.90},
    "Baker Hughes":   {"base": 3,  "oil_beta": 0.85},
    "Hapag-Lloyd":    {"base": 5,  "oil_beta": 0.10},
    "Maersk":         {"base": 6,  "oil_beta": 0.12},
}

def estimate_earnings(base, oil_beta):
    oil_effect = base * oil_beta * (brent_price / 70)
    asia_effect = base * 0.25 * (asia_trade_index - 1)
    noise = np.random.normal(0, base * volatility)

    earnings = base + oil_effect + asia_effect + noise
    return max(earnings, 0)

# --------------------------------
# Run Earnings Simulation
# --------------------------------
results = []

for name, params in companies.items():
    earnings_est = estimate_earnings(params["base"], params["oil_beta"])

    results.append({
        "company": name,
        "earnings_usd_billion_est": round(earnings_est, 2)
    })

df = pd.DataFrame(results)

# --------------------------------
# Sector Summary
# --------------------------------
sector_summary = {
    "total_estimated_earnings_usd_billion":
        round(df["earnings_usd_billion_est"].sum(), 2),

    "brent_proxy_used":
        brent_price,

    "asia_trade_index_used":
        asia_trade_index
}

print("\nEstimated Earnings (USD Billions)\n")
print(df.sort_values(by="earnings_usd_billion_est", ascending=False))

print("\nSector Summary\n")
for k, v in sector_summary.items():
    print(f"{k}: {v}")

# Optional persistence
prices.to_csv("market_prices.csv")
df.to_csv("earnings_estimates.csv")

print("\nSaved:")
print(" - market_prices.csv")
print(" - earnings_estimates.csv")

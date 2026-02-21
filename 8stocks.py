import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Original 5 stocks + new 3
# -----------------------------
# Latest prices in local currency / USD
stocks = {
    "JPM": {"price": 302.55, "dividend": 5.80},          # USD
    "SK hynix": {"price": 880000, "dividend": 2429},     # KRW
    "TSMC": {"price": 366.36, "dividend": 3.12},         # USD
    "Dell": {"price": 117.49, "dividend": 2.10},         # USD
    "IBM": {"price": 262.38, "dividend": 6.72},          # USD
    "SoftBank": {"price": 8000, "dividend": 50},         # JPY
    "NVIDIA": {"price": 400, "dividend": 0.25},          # USD
    "Samsung Electronics": {"price": 65000, "dividend": 1300} # KRW
}

# Growth rates (CAGR assumptions)
growth_rates = {
    "JPM": 0.07,
    "SK hynix": 0.10,
    "TSMC": 0.12,
    "Dell": 0.05,
    "IBM": 0.03,
    "SoftBank": 0.08,
    "NVIDIA": 0.25,
    "Samsung Electronics": 0.10
}

# Currency conversion to USD (approximate)
currency_rates = {
    "USD": 1,
    "KRW": 0.00075,  # 1 KRW ≈ 0.00075 USD
    "JPY": 0.0071    # 1 JPY ≈ 0.0071 USD
}

# Map stocks to currency
stock_currency = {
    "JPM": "USD",
    "SK hynix": "KRW",
    "TSMC": "USD",
    "Dell": "USD",
    "IBM": "USD",
    "SoftBank": "JPY",
    "NVIDIA": "USD",
    "Samsung Electronics": "KRW"
}

shares_held = 1000
years = [1,2,3]

# Prepare DataFrame
portfolio = pd.DataFrame(columns=["Stock", "Year", "PriceUSD", "DividendsUSD", "TotalReturnUSD"])

for stock, data in stocks.items():
    price_local = data["price"]
    dividend_local = data["dividend"]
    growth = growth_rates[stock]
    currency = stock_currency[stock]
    rate = currency_rates[currency]
    
    # Convert price/dividends to USD
    price = price_local * rate
    dividend = dividend_local * rate
    
    for year in years:
        projected_price = price * ((1 + growth) ** year)
        total_dividends = dividend * shares_held * year
        total_return = (projected_price - price) * shares_held + total_dividends
        
        portfolio = pd.concat([portfolio, pd.DataFrame([{
            "Stock": stock,
            "Year": year,
            "PriceUSD": round(projected_price,2),
            "DividendsUSD": round(total_dividends,2),
            "TotalReturnUSD": round(total_return,2)
        }])], ignore_index=True)

# Display portfolio table
print("\nPortfolio Simulation (Projected Dividend Income + Total Return in USD):")
print(portfolio)

# -----------------------------
# Plot Total Return
# -----------------------------
plt.figure(figsize=(12,6))
for stock in stocks.keys():
    subset = portfolio[portfolio["Stock"] == stock]
    plt.plot(subset["Year"], subset["TotalReturnUSD"], marker="o", label=stock)
plt.title("Projected Total Return for 1000 Shares (1–3 Years) USD")
plt.xlabel("Year")
plt.ylabel("Total Return (USD)")
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------
# Plot Dividend Income
# -----------------------------
plt.figure(figsize=(12,6))
for stock in stocks.keys():
    subset = portfolio[portfolio["Stock"] == stock]
    plt.plot(subset["Year"], subset["DividendsUSD"], marker="s", label=stock)
plt.title("Projected Dividend Income for 1000 Shares (1–3 Years) USD")
plt.xlabel("Year")
plt.ylabel("Dividend Income (USD)")
plt.grid(True)
plt.legend()
plt.show()

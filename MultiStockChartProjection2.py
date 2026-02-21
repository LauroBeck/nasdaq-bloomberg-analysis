import pandas as pd
import matplotlib.pyplot as plt

# Define stocks, latest prices, annual dividend per share (from your MultiStockChartProjection)
stocks = {
    "JPM": {"price": 302.55, "dividend": 5.80},
    "SK hynix": {"price": 880000, "dividend": 2429},  # KRW
    "TSMC": {"price": 366.36, "dividend": 3.12},
    "Dell": {"price": 117.49, "dividend": 2.10},
    "IBM": {"price": 262.38, "dividend": 6.72}
}

shares_held = 1000
years = [1, 2, 3]

# Assume simple annual price growth rates (%) based on historical trend
growth_rates = {
    "JPM": 0.07,       # 7% per year
    "SK hynix": 0.10,  # 10% per year
    "TSMC": 0.12,      # 12% per year
    "Dell": 0.05,      # 5% per year
    "IBM": 0.03        # 3% per year
}

# Prepare DataFrame to store simulation results
portfolio = pd.DataFrame(columns=["Stock", "Year", "Price", "Dividends", "TotalReturn"])

for stock, data in stocks.items():
    price = data["price"]
    dividend = data["dividend"]
    growth = growth_rates[stock]
    
    for year in years:
        projected_price = price * ((1 + growth) ** year)
        total_dividends = dividend * shares_held * year
        total_return = (projected_price - price) * shares_held + total_dividends
        portfolio = pd.concat([portfolio, pd.DataFrame([{
            "Stock": stock,
            "Year": year,
            "Price": projected_price,
            "Dividends": total_dividends,
            "TotalReturn": total_return
        }])], ignore_index=True)

# Round numbers for readability
portfolio[["Price", "Dividends", "TotalReturn"]] = portfolio[["Price", "Dividends", "TotalReturn"]].round(2)

print("\nPortfolio Simulation (Projected Dividend Income + Total Return):")
print(portfolio)

# Plot Total Return over 3 years
plt.figure(figsize=(12,6))
for stock in stocks.keys():
    subset = portfolio[portfolio["Stock"] == stock]
    plt.plot(subset["Year"], subset["TotalReturn"], marker="o", label=stock)
plt.title("Projected Total Return for 1000 Shares (1–3 Years)")
plt.xlabel("Year")
plt.ylabel("Total Return ($)")
plt.grid(True)
plt.legend()
plt.show()

# Plot Dividend Income over 3 years
plt.figure(figsize=(12,6))
for stock in stocks.keys():
    subset = portfolio[portfolio["Stock"] == stock]
    plt.plot(subset["Year"], subset["Dividends"], marker="s", label=stock)
plt.title("Projected Dividend Income for 1000 Shares (1–3 Years)")
plt.xlabel("Year")
plt.ylabel("Dividend Income ($)")
plt.grid(True)
plt.legend()
plt.show()

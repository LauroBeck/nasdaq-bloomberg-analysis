import yfinance as yf

tickers = ["AAPL", "GOOGL"]

# Download data for the tickers
data = yf.download(tickers, start="2023-01-01", end="2024-01-01")

# Print the available columns to see what's being returned
print(data.head())

# Access 'Adj Close' only if it exists
if 'Adj Close' in data.columns:
    data = data['Adj Close']
else:
    print("Column 'Adj Close' not found. Available columns are:", data.columns)

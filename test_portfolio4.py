import yfinance as yf
import pandas as pd

# Index mapping from the provided image
image_indices = {
    "Index": ["Nasdaq Composite", "Dow Jones Industrial Average", "S&P 500", "Russell 2000 Index", "NYSE Composite"],
    "Ticker": ["^IXIC", "^DJI", "^GSPC", "^RUT", "^NYA"],
    "Image_Value": [23066.47, 50121.40, 6941.47, 2669.47, 23479.72]
}

# Create a reference DataFrame for the image data
df_image = pd.DataFrame(image_indices)
print("--- Data from Provided Image (Feb 11, 2026) ---")
print(df_image)
print("\n")

# 1. Fetch data for Feb 11, 2026 (matching the image date)
# Note: 'end' date must be the following day
start_date = "2026-02-11"
end_date = "2026-02-12"

print(f"--- Fetching historical data for {start_date} ---")
data = yf.download(image_indices["Ticker"], start=start_date, end=end_date)

# 2. Extract closing prices
# Modern yfinance versions typically merge 'Adj Close' into 'Close' by default
if 'Close' in data.columns:
    fetched_prices = data['Close'].iloc[0]
else:
    fetched_prices = data['Adj Close'].iloc[0]

# 3. Compare image values with fetched values
for _, row in df_image.iterrows():
    ticker = row['Ticker']
    name = row['Index']
    img_val = row['Image_Value']
    
    try:
        actual_val = fetched_prices[ticker]
        diff = actual_val - img_val
        print(f"{name} ({ticker}):")
        print(f"  - Image: {img_val:,.2f}")
        print(f"  - Fetched: {actual_val:,.2f} (Diff: {diff:,.2f})")
    except KeyError:
        print(f"Data for {ticker} not available for this date range.")

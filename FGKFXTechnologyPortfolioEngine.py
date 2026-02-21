import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# 1. SETUP DATA FROM BLOOMBERG IMAGE
# Mapping the "Identifier" and "Avg % Wgt Port" columns from the screenshot
data = {
    "Company": [
        "NVIDIA CORP", "APPLE INC", "SANDISK CORP", "MICROSOFT CORP", 
        "BROADCOM INC", "CIENA CORP", "ORACLE CORP", "PURE STORAGE INC", 
        "SALESFORCE INC", "CLOUDFLARE INC", "SERVICENOW INC", "ADVANCED MICRO DEVICES"
    ],
    "Ticker": ["NVDA", "AAPL", "SNDK", "MSFT", "AVGO", "CIEN", "ORCL", "PSTG", "CRM", "NET", "NOW", "AMD"],
    "Avg_Wgt_Port": [15.33, 7.83, 0.29, 5.63, 1.35, 0.58, 1.90, 1.34, 1.33, 0.57, 0.86, 0.48],
    "Image_Tot_Rtn": [879.28, 82.53, 1129.99, 77.78, 492.33, 384.06, 92.91, 140.29, 27.94, 235.19, 28.55, 215.01]
}

df = pd.DataFrame(data)

# 2. FETCH LIVE DATA (To compare image performance with current metrics)
print("Fetching current market data for FGKFX holdings...")
tickers = df["Ticker"].tolist()
# 'SNDK' is used for the spun-off SanDisk Corp as of 2025/2026
current_data = yf.download(tickers, period="1y")["Close"].iloc[[-1]].T
current_data.columns = ["Current_Price"]
current_data.index.name = "Ticker"

# 3. CALCULATE PORTFOLIO METRICS
# Calculating the Contribution to Return (CTR) as seen in the Bloomberg 'CTR Port' column
# CTR = (Weight * Return) / 100
df["CTR_Port"] = (df["Avg_Wgt_Port"] * df["Image_Tot_Rtn"]) / 100

# 4. OUTPUT RESULTS
print("\n--- FGKFX Information Technology Sector Analysis ---")
print(df[["Company", "Avg_Wgt_Port", "Image_Tot_Rtn", "CTR_Port"]].to_string(index=False))

# Summary stats to match the 'Information Technology' row in the image
total_weight = df["Avg_Wgt_Port"].sum()
total_ctr = df["CTR_Port"].sum()
print(f"\nTotal IT Sector Weight: {total_weight:.2f}% (Image: 46.34%)")
print(f"Total IT Sector CTR:    {total_ctr:.2f} (Image: 96.78)")

# 5. VISUALIZE CONTRIBUTION
plt.style.use("dark_background")
df.sort_values("CTR_Port", ascending=True).plot(
    kind='barh', x='Company', y='CTR_Port', color='orange', figsize=(10,6)
)
plt.title("Contribution to Return (CTR) by Holding")
plt.xlabel("CTR (Basis Points/Percentage Points)")
plt.show()

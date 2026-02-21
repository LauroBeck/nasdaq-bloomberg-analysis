import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

# -----------------------------
# SETTINGS
# -----------------------------
tickers_main = {
    "Coinbase Global Inc": "COIN",
    "Pinterest Inc": "PINS"
}

market_tickers = {
    "DOW": "^DJI",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "COPPER": "HG=F"
}

start_date = "2023-01-01"
end_date = "2024-01-01"

# -----------------------------
# DOWNLOAD DATA
# -----------------------------
all_tickers = list(tickers_main.values()) + list(market_tickers.values())
data = yf.download(all_tickers, start=start_date, end=end_date, group_by="ticker")

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_price_info(df):
    close = df["Close"]
    latest = close.iloc[-1]
    prev = close.iloc[-2]
    change = latest - prev
    pct_change = (change / prev) * 100
    return latest, change, pct_change

def plot_small_chart(ax, series, title):
    color = "lime" if series.iloc[-1] >= series.iloc[0] else "red"
    ax.plot(series, color=color, linewidth=1.5)
    ax.set_title(title, color="white", fontsize=8)
    ax.set_facecolor("#111111")
    ax.tick_params(colors="white", labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("white")

# -----------------------------
# BUILD FIGURE LAYOUT
# -----------------------------
plt.style.use("dark_background")
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("black")

gs = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[2, 2, 1, 1])

# -----------------------------
# MAIN STOCK PANEL
# -----------------------------
main_stock_name = list(tickers_main.keys())[0]
main_stock_symbol = tickers_main[main_stock_name]
main_df = data[main_stock_symbol]

ax_main = fig.add_subplot(gs[:, :2])
ax_main.set_facecolor("#111111")

ax_main.plot(main_df["Close"], color="red", linewidth=2)
ax_main.set_title(f"{main_stock_name} - 1 Year", fontsize=16, color="white")
ax_main.tick_params(colors="white")

latest, change, pct = get_price_info(main_df)

ax_main.text(
    0.02, 0.95,
    f"Last: {latest:.2f}\nChange: {change:.2f} ({pct:.2f}%)",
    transform=ax_main.transAxes,
    fontsize=12,
    color="red" if change < 0 else "lime",
    verticalalignment="top"
)

# -----------------------------
# SECONDARY STOCK PANEL
# -----------------------------
second_stock_name = list(tickers_main.keys())[1]
second_stock_symbol = tickers_main[second_stock_name]
second_df = data[second_stock_symbol]

ax_secondary = fig.add_subplot(gs[0, 2])
plot_small_chart(ax_secondary, second_df["Close"], second_stock_name)

# -----------------------------
# MARKET SIDEBAR PANELS
# -----------------------------
row = 1
col = 2
for name, symbol in market_tickers.items():
    df = data[symbol]
    ax = fig.add_subplot(gs[row, col])
    plot_small_chart(ax, df["Close"], name)

    col += 1
    if col > 3:
        col = 2
        row += 1
        if row > 2:
            break

# -----------------------------
# BREAKING NEWS STYLE TEXT
# -----------------------------
fig.text(
    0.05, 0.05,
    "BREAKING NEWS: Market volatility continues amid tech earnings.\n"
    "Crypto stocks decline as digital assets retrace recent gains.\n"
    "Federal Reserve comments impact bond and equity markets.",
    color="red",
    fontsize=12,
    weight="bold"
)

# -----------------------------
# BLOOMBERG HEADER
# -----------------------------
fig.text(0.02, 0.95, "Bloomberg", fontsize=28, color="white", weight="bold")

plt.tight_layout(rect=[0, 0.08, 1, 0.93])
plt.show()

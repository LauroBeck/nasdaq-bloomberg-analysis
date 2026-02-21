# ==========================================
# BLOOMBERG STYLE MARKET DASHBOARD
# Production Safe Version
# ==========================================

import os
import matplotlib

# Use non-interactive backend if no display (Linux/Server safe)
if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

# ==========================================
# SETTINGS
# ==========================================

MAIN_STOCK = "COIN"
SECONDARY_STOCK = "PINS"

MARKET_TICKERS = {
    "DOW": "^DJI",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "COPPER": "HG=F"
}

START_DATE = "2023-01-01"
END_DATE = "2024-01-01"

OUTPUT_FILE = "bloomberg_dashboard.png"

# ==========================================
# DOWNLOAD DATA
# ==========================================

all_tickers = [MAIN_STOCK, SECONDARY_STOCK] + list(MARKET_TICKERS.values())

print("Downloading market data...")
data = yf.download(
    all_tickers,
    start=START_DATE,
    end=END_DATE,
    group_by="ticker",
    progress=False
)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_price_info(df):
    close = df["Close"].dropna()
    if len(close) < 2:
        return 0, 0, 0
    latest = close.iloc[-1]
    prev = close.iloc[-2]
    change = latest - prev
    pct_change = (change / prev) * 100
    return latest, change, pct_change


def plot_chart(ax, series, title, big=False):
    series = series.dropna()

    if len(series) < 2:
        ax.set_title(f"{title} (No Data)", color="white")
        return

    color = "#00ff99" if series.iloc[-1] >= series.iloc[0] else "#ff3b3b"

    ax.plot(series, color=color, linewidth=2 if big else 1)
    ax.set_title(title, color="white", fontsize=14 if big else 8)
    ax.set_facecolor("#111111")
    ax.tick_params(colors="white", labelsize=8 if big else 6)

    for spine in ax.spines.values():
        spine.set_color("#333333")


# ==========================================
# BUILD FIGURE
# ==========================================

plt.style.use("dark_background")

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("black")

gs = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[2, 2, 1, 1])

# ==========================================
# MAIN STOCK PANEL
# ==========================================

main_df = data[MAIN_STOCK]
ax_main = fig.add_subplot(gs[:, :2])

plot_chart(ax_main, main_df["Close"], f"{MAIN_STOCK} - 1 Year", big=True)

latest, change, pct = get_price_info(main_df)

ax_main.text(
    0.02, 0.95,
    f"Last: {latest:.2f}\nChange: {change:.2f} ({pct:.2f}%)",
    transform=ax_main.transAxes,
    fontsize=14,
    color="#00ff99" if change > 0 else "#ff3b3b",
    verticalalignment="top",
    fontweight="bold"
)

# ==========================================
# SECONDARY STOCK PANEL
# ==========================================

ax_secondary = fig.add_subplot(gs[0, 2])
plot_chart(ax_secondary, data[SECONDARY_STOCK]["Close"], SECONDARY_STOCK)

# ==========================================
# MARKET SIDEBAR
# ==========================================

row = 1
col = 2

for name, symbol in MARKET_TICKERS.items():
    if row > 2:
        break

    ax = fig.add_subplot(gs[row, col])
    plot_chart(ax, data[symbol]["Close"], name)

    col += 1
    if col > 3:
        col = 2
        row += 1

# ==========================================
# BREAKING NEWS BANNER
# ==========================================

fig.text(
    0.03, 0.05,
    "BREAKING NEWS: Market volatility rises amid earnings season.\n"
    "Crypto-linked equities face renewed pressure.\n"
    "Federal Reserve policy outlook remains key market driver.",
    color="#ff3b3b",
    fontsize=12,
    weight="bold"
)

# ==========================================
# HEADER
# ==========================================

fig.text(0.02, 0.95, "Bloomberg", fontsize=28, color="white", weight="bold")
fig.text(0.80, 0.95, datetime.now().strftime("%Y-%m-%d"), fontsize=12, color="white")

# ==========================================
# FINALIZE LAYOUT
# ==========================================

plt.tight_layout(rect=[0, 0.08, 1, 0.93])

# ==========================================
# SAVE OUTPUT (ALWAYS WORKS)
# ==========================================

plt.savefig(OUTPUT_FILE, dpi=300, facecolor="black", bbox_inches="tight")
print(f"Dashboard saved as {OUTPUT_FILE}")

# ==========================================
# SHOW ONLY IF INTERACTIVE BACKEND
# ==========================================

backend = matplotlib.get_backend().lower()
interactive_backends = ["tkagg", "qt5agg", "qtagg", "macosx"]

if backend in interactive_backends:
    plt.show()

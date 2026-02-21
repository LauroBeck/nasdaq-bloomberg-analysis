# ==========================================================
# BLOOMBERG MARKETS MENU STYLE DASHBOARD
# FULL DEPLOYABLE VERSION - test_portfolio9.py
# ==========================================================

import os
import matplotlib

# Safe backend detection (server compatible)
if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# ==========================================================
# SETTINGS
# ==========================================================

TICKERS = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "FTSE 100": "^FTSE",
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
}

START_DATE = "2024-01-01"
OUTPUT_FILE = "bloomberg_markets_menu.png"

# ==========================================================
# DOWNLOAD DATA
# ==========================================================

print("Downloading market data...")
data = yf.download(
    list(TICKERS.values()),
    start=START_DATE,
    progress=False,
    group_by="ticker"
)

# ==========================================================
# HELPER FUNCTION
# ==========================================================

def get_price(symbol):
    df = data[symbol]["Close"].dropna()
    if len(df) < 2:
        return 0, 0

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    pct = ((latest - prev) / prev) * 100
    return latest, pct


# ==========================================================
# BUILD FIGURE
# ==========================================================

plt.style.use("dark_background")

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("#000000")

ax = fig.add_subplot(111)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# ==========================================================
# HEADER
# ==========================================================

ax.text(5, 92, "Bloomberg", fontsize=36, fontweight="bold")
ax.text(5, 86, "Markets", fontsize=18, color="orange")
ax.text(80, 92, datetime.now().strftime("%Y-%m-%d %H:%M"), fontsize=12)

# ==========================================================
# LEFT COLUMN - NEWS
# ==========================================================

ax.text(5, 75, "News", fontsize=16, color="#aaaaaa")

news_items = [
    "Deals",
    "Fixed Income",
    "ETFs",
    "FX",
    "Alt. Investing",
    "Economic Calendar"
]

y_pos = 70
for item in news_items:
    ax.text(5, y_pos, item, fontsize=13)
    y_pos -= 5

# ==========================================================
# CENTER COLUMN - DATA
# ==========================================================

ax.text(30, 75, "Data", fontsize=16, color="#aaaaaa")

data_items = [
    "Stocks",
    "Futures",
    "Rates & Bonds",
    "Currencies",
    "Commodities",
    "Sectors",
    "Energy"
]

y_pos = 70
for item in data_items:
    ax.text(30, y_pos, item, fontsize=13)
    y_pos -= 5

# ==========================================================
# RIGHT COLUMN - TOP SECURITIES GRID
# ==========================================================

ax.text(55, 75, "Top Securities", fontsize=16, color="#aaaaaa")

x_positions = [55, 70, 85]
y_positions = [65, 50, 35]

i = 0
for name, symbol in TICKERS.items():
    col = i % 3
    row = i // 3

    if row >= 3:
        break

    x = x_positions[col]
    y = y_positions[row]

    latest, pct = get_price(symbol)
    color = "#00ff99" if pct >= 0 else "#ff3b3b"
    arrow = "▲" if pct >= 0 else "▼"

    # Card background
    rect = patches.FancyBboxPatch(
        (x, y),
        12,
        10,
        boxstyle="round,pad=0.5",
        facecolor="#1a1a1a",
        edgecolor="#333333"
    )
    ax.add_patch(rect)

    ax.text(x + 1, y + 7, name, fontsize=10, fontweight="bold")
    ax.text(x + 1, y + 4, f"{latest:,.2f}", fontsize=10)
    ax.text(x + 1, y + 1.5, f"{arrow} {pct:.2f}%", fontsize=10, color=color)

    i += 1

# ==========================================================
# FOOTER LINE
# ==========================================================

ax.text(
    5,
    5,
    "Bloomberg Markets Dashboard Replica | Data via Yahoo Finance",
    fontsize=10,
    color="#666666"
)

# ==========================================================
# SAVE + SAFE SHOW
# ==========================================================

plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300, facecolor="#000000", bbox_inches="tight")
print(f"Dashboard saved as {OUTPUT_FILE}")

backend = matplotlib.get_backend().lower()
interactive_backends = ["tkagg", "qt5agg", "qtagg", "macosx"]

if backend in interactive_backends:
    plt.show()

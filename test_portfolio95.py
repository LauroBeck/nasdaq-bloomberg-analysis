# ==========================================================
# BLOOMBERG MARKETS - FULL GRAPHICAL VERSION
# ==========================================================

import os
import matplotlib

# Safe backend (server compatible)
if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
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
OUTPUT_FILE = "bloomberg_markets_graphical.png"

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

fig = plt.figure(figsize=(20, 11))
ax = fig.add_subplot(111)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# ---------- Gradient Background ----------
gradient = np.linspace(0.05, 0.15, 256)
gradient = np.vstack((gradient, gradient))
ax.imshow(
    gradient,
    extent=[0, 100, 0, 100],
    aspect="auto",
    cmap="gray",
    alpha=0.25
)

# ==========================================================
# HEADER
# ==========================================================

ax.text(5, 94, "Bloomberg", fontsize=38, fontweight="bold")
ax.text(5, 88, "Markets", fontsize=18, color="#f5a623", fontweight="bold")
ax.text(80, 94, datetime.now().strftime("%Y-%m-%d %H:%M"),
        fontsize=11, color="#aaaaaa")

# Divider
ax.plot([5, 95], [85, 85], color="#222222", linewidth=1)

# ==========================================================
# LEFT COLUMN
# ==========================================================

ax.text(5, 78, "News", fontsize=16, color="#bbbbbb", fontweight="bold")

news_items = [
    "Deals",
    "Fixed Income",
    "ETFs",
    "FX",
    "Alt. Investing",
    "Economic Calendar"
]

y = 73
for item in news_items:
    ax.text(5, y, item, fontsize=13, color="#dddddd")
    y -= 5

# ==========================================================
# CENTER COLUMN
# ==========================================================

ax.text(30, 78, "Data", fontsize=16, color="#bbbbbb", fontweight="bold")

data_items = [
    "Stocks",
    "Futures",
    "Rates & Bonds",
    "Currencies",
    "Commodities",
    "Sectors",
    "Energy"
]

y = 73
for item in data_items:
    ax.text(30, y, item, fontsize=13, color="#dddddd")
    y -= 5

# ==========================================================
# RIGHT COLUMN - TOP SECURITIES
# ==========================================================

ax.text(55, 78, "Top Securities", fontsize=16,
        color="#bbbbbb", fontweight="bold")

x_positions = [55, 72, 89]
y_positions = [68, 52, 36]

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

    # Glass Card Effect
    rect = patches.FancyBboxPatch(
        (x - 1, y - 2),
        14,
        12,
        boxstyle="round,pad=0.6",
        facecolor="#1c1c1e",
        edgecolor="#333333",
        linewidth=1.2
    )
    ax.add_patch(rect)

    ax.text(x, y + 7, name, fontsize=11,
            fontweight="bold", color="#ffffff")

    ax.text(x, y + 3.5, f"{latest:,.2f}",
            fontsize=12, color="#dddddd")

    ax.text(x, y + 1,
            f"{arrow} {pct:.2f}%",
            fontsize=11,
            color=color,
            fontweight="bold")

    i += 1

# ==========================================================
# FOOTER
# ==========================================================

ax.plot([5, 95], [10, 10], color="#222222", linewidth=1)

ax.text(
    5,
    5,
    "Bloomberg Markets Dashboard Replica  |  Data via Yahoo Finance",
    fontsize=9,
    color="#666666"
)

# ==========================================================
# SAVE + SAFE SHOW
# ==========================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    facecolor="#000000",
    bbox_inches="tight"
)

print(f"Dashboard saved as {OUTPUT_FILE}")

backend = matplotlib.get_backend().lower()
interactive_backends = ["tkagg", "qt5agg", "qtagg", "macosx"]

if backend in interactive_backends:
    plt.show()

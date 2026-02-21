import yfinance as yf
import pandas as pd
import numpy as np

# ------------------------------------------------
# CONFIGURATION (Bloomberg-Screen Consistent)
# ------------------------------------------------

stocks = {
    "Pilbara Minerals (PLS)": "PLS.AX",
    "Rio Tinto (RIO)": "RIO.AX",
    "BHP Group (BHP)": "BHP.AX",
    "BlueScope Steel (BSL)": "BSL.AX"
}

materials_index = "^AXMJ"

lookback = "6mo"

# ------------------------------------------------
# DOWNLOAD DATA
# ------------------------------------------------

tickers = list(stocks.values()) + [materials_index]

print("\nDownloading market data...\n")

data = yf.download(tickers, period=lookback)

prices = data["Close"]
returns = prices.pct_change().dropna()

# ------------------------------------------------
# SECTOR REGIME DETECTION
# ------------------------------------------------

index_momentum = prices[materials_index].pct_change(126).iloc[-1]

if index_momentum > 0.10:
    regime = "STRONG MATERIALS SECTOR"
elif index_momentum < -0.10:
    regime = "WEAK MATERIALS SECTOR"
else:
    regime = "NEUTRAL SECTOR"

# ------------------------------------------------
# FACTOR CALCULATIONS
# ------------------------------------------------

metrics = {}

for name, ticker in stocks.items():

    stock_ret = returns[ticker]
    index_ret = returns[materials_index]

    beta = np.cov(stock_ret, index_ret)[0, 1] / np.var(index_ret)

    momentum = prices[ticker].pct_change(126).iloc[-1]
    volatility = stock_ret.std() * np.sqrt(252)

    relative_strength = (
        prices[ticker].iloc[-1] /
        prices[materials_index].iloc[-1]
    )

    metrics[name] = {
        "price": prices[ticker].iloc[-1],
        "momentum_6m": momentum,
        "beta_vs_sector": beta,
        "volatility": volatility,
        "relative_strength": relative_strength
    }

df = pd.DataFrame(metrics).T

# ------------------------------------------------
# DECISION ENGINE (Bloomberg-like logic)
# ------------------------------------------------

decisions = {}

for stock in df.index:

    m = df.loc[stock, "momentum_6m"]
    beta = df.loc[stock, "beta_vs_sector"]
    vol = df.loc[stock, "volatility"]

    decision = "HOLD"

    if regime.startswith("STRONG") and m > 0.08:
        decision = "BUY"

    if regime.startswith("STRONG") and m > 0.15 and beta > 1:
        decision = "STRONG BUY"

    elif m < -0.10:
        decision = "SELL"

    elif m < -0.20:
        decision = "STRONG SELL"

    if vol > df["volatility"].mean():
        decision += " (High Risk)"

    decisions[stock] = decision

# ------------------------------------------------
# OUTPUT
# ------------------------------------------------

print("\n================ MINING MODEL v2.0 ================\n")

print(f"Materials Sector Regime: {regime}")
print(f"Materials Index Momentum (6M): {index_momentum:.2%}\n")

for stock in df.index:

    row = df.loc[stock]

    print(stock)
    print(f"  Price: {row['price']:.2f}")
    print(f"  Momentum (6M): {row['momentum_6m']:.2%}")
    print(f"  Beta vs Sector: {row['beta_vs_sector']:.2f}")
    print(f"  Volatility: {row['volatility']:.2%}")
    print(f"  Relative Strength: {row['relative_strength']:.6f}")
    print(f"  Decision: {decisions[stock]}")
    print("")

print("===================================================\n")

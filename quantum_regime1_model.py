import numpy as np
import pandas as pd
import yfinance as yf

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

# -----------------------------------
# CONFIGURATION
# -----------------------------------

TICKER = "JPMC34.SA"     # Example B3 BDR
PERIOD = "1y"
INTERVAL = "1d"
TRADING_DAYS = 252
SHOTS = 4000

# -----------------------------------
# DATA DOWNLOAD
# -----------------------------------

print(f"Downloading data for {TICKER}...")

df = yf.download(TICKER, period=PERIOD, interval=INTERVAL)
df = df.dropna()

if df.empty:
    raise RuntimeError("No market data retrieved.")

close = df["Close"]

# yfinance safety (sometimes multi-index)
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

close = close.astype(float)

# -----------------------------------
# CLASSICAL MARKET METRICS
# -----------------------------------

returns = close.pct_change().dropna()

momentum_3m = float(close.iloc[-1] / close.iloc[-63] - 1)
volatility = float(returns.std() * np.sqrt(TRADING_DAYS))

print("\nCLASSICAL MARKET STATE")
print("-----------------------------------")
print(f"Last Price:  {close.iloc[-1]:.2f} BRL")
print(f"3M Momentum: {momentum_3m:.2%}")
print(f"Volatility:  {volatility:.2%}")

# -----------------------------------
# NORMALIZATION → QUANTUM ANGLES
# -----------------------------------

def normalize_to_angle(x, clip=1.0):
    """
    Maps variable into [0, π]
    Stable for financial metrics
    """
    x = np.clip(x, -clip, clip)
    return (x + clip) / (2 * clip) * np.pi

theta_regime = normalize_to_angle(volatility)
theta_momentum = normalize_to_angle(momentum_3m)

# -----------------------------------
# QUANTUM CIRCUIT
# -----------------------------------

qc = QuantumCircuit(3)

# Encode market uncertainty
qc.ry(theta_regime, 0)

# Encode directional bias
qc.ry(theta_momentum, 1)

# Dependency structure (factor coupling)
qc.cx(0, 1)
qc.cx(0, 2)

qc.measure_all()

# -----------------------------------
# EXECUTION (Aer Simulator)
# -----------------------------------

backend = Aer.get_backend("aer_simulator")

compiled = transpile(qc, backend)

job = backend.run(compiled, shots=SHOTS)
result = job.result()

counts = result.get_counts()

# -----------------------------------
# PROBABILITY EXTRACTION
# -----------------------------------

total = sum(counts.values())

def probability(state: str) -> float:
    return counts.get(state, 0) / total

p_calm = probability("000")
p_stress = probability("111")

print("\nQUANTUM STATE DISTRIBUTION")
print("-----------------------------------")
for state, c in sorted(counts.items()):
    print(f"{state}: {c}")

print("\nREGIME PROBABILITIES")
print("-----------------------------------")
print(f"Calm Regime   (000): {p_calm:.2%}")
print(f"Stress Regime (111): {p_stress:.2%}")

# -----------------------------------
# DECISION HEURISTIC (toy logic)
# -----------------------------------

if p_stress > 0.6:
    regime_view = "RISK OFF"
elif p_calm > 0.6:
    regime_view = "RISK ON"
else:
    regime_view = "NEUTRAL"

print(f"\nQuantum Regime View: {regime_view}")

print("\nDone.")

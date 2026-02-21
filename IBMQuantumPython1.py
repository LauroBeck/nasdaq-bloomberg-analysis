import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile

# -----------------------------
# MARKET INPUTS (from screen)
# -----------------------------
spx = 0.17
dow = 0.11
ndx = 0.13

mean_move = np.mean([spx, dow, ndx])
dispersion = np.std([spx, dow, ndx])

# Normalize helpers
def norm(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)

# Map to angles
theta_direction = norm(mean_move, -1, 1) * np.pi
theta_agreement = (1 - norm(dispersion, 0, 1)) * np.pi
theta_volatility = norm(dispersion, 0, 1) * np.pi

# -----------------------------
# QUANTUM CIRCUIT
# -----------------------------
qc = QuantumCircuit(3, 3)

qc.ry(theta_direction, 0)     # Market bias
qc.ry(theta_agreement, 1)     # Coherence
qc.ry(theta_volatility, 2)    # Instability

# Entanglement → cross-market coupling
qc.cx(0, 1)
qc.cx(1, 2)

qc.measure([0, 1, 2], [0, 1, 2])

print(qc.draw())

# -----------------------------
# SIMULATION
# -----------------------------
simulator = AerSimulator()
compiled = transpile(qc, simulator)

job = simulator.run(compiled, shots=2000)
counts = job.result().get_counts()

total = sum(counts.values())

print("\nProbabilities:")
for state, c in counts.items():
    print(state, round(c / total, 3))

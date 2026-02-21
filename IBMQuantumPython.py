import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit import transpile

# -----------------------------
# CLASSICAL MARKET INPUTS
# -----------------------------
momentum = 0.65      # Example normalized signal (0 → 1)
volatility = 0.30
macro = 0.55

# Convert to rotation angles
theta_momentum = momentum * np.pi
theta_volatility = volatility * np.pi
theta_macro = macro * np.pi

# -----------------------------
# QUANTUM CIRCUIT
# -----------------------------
qc = QuantumCircuit(3, 3)

# Encode features into qubits
qc.ry(theta_momentum, 0)
qc.ry(theta_volatility, 1)
qc.ry(theta_macro, 2)

# Entanglement → interaction between factors
qc.cx(0, 1)
qc.cx(1, 2)

# Measurement
qc.measure([0, 1, 2], [0, 1, 2])

print(qc.draw())

# -----------------------------
# SIMULATION
# -----------------------------
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)

job = simulator.run(compiled_circuit, shots=2000)
result = job.result()
counts = result.get_counts()

print("\nProbabilities:")
total = sum(counts.values())
for state, count in counts.items():
    print(f"{state}: {count / total:.3f}")

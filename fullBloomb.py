import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit_aer import AerSimulator
from qiskit_addon_sqd.qubit import solve_qubit

# ============================================================
# Portfolio Optimization Parameters
# ============================================================

n = 8                  # number of assets / qubits
shots = 20000
t = 3.0                # evolution time
max_eigvals = 5
max_subspace = 500

np.random.seed(42)
mu = np.random.uniform(0.05, 0.2, size=n)          # expected returns
sigma = np.random.uniform(0.01, 0.05, size=(n,n))  # risk covariance
sigma = (sigma + sigma.T)/2                         # symmetric
np.fill_diagonal(sigma, 0.05)

lambda_risk = 0.5  # risk vs return tradeoff

# ============================================================
# Build Hamiltonian for portfolio
# ============================================================

pauli_list = []

def zz_term(i,j,coeff):
    label = ["I"]*n
    label[i] = "Z"
    label[j] = "Z"
    return ("".join(label), coeff)

def z_term(i, coeff):
    label = ["I"]*n
    label[i] = "Z"
    return ("".join(label), coeff)

# Risk terms
for i in range(n):
    for j in range(i+1,n):
        pauli_list.append(zz_term(i,j, sigma[i,j]))

# Return terms
for i in range(n):
    pauli_list.append(z_term(i, -lambda_risk*mu[i]))

H = SparsePauliOp.from_list(pauli_list).simplify()
print("Hamiltonian for portfolio built.")

# ============================================================
# Build Ansatz Circuit
# ============================================================

qc = QuantumCircuit(n)
qc.h(range(n))  # Hadamard prep for superposition

evo_gate = PauliEvolutionGate(H, time=t, synthesis=LieTrotter(reps=1))
qc.append(evo_gate, range(n))
qc = qc.decompose(reps=5)

# Measurement
creg = ClassicalRegister(n)
qc.add_register(creg)
qc.measure(range(n), range(n))

# ============================================================
# Run Simulation
# ============================================================

backend = AerSimulator()
qc = transpile(qc, backend, optimization_level=1)

print("Running circuit...")
job = backend.run(qc, shots=shots)
result = job.result()
counts = result.get_counts()

bitstrings = list(counts.keys())
bitmatrix = np.array([[int(b) for b in bs] for bs in bitstrings], dtype=bool)

# Downsample if too large
if len(bitstrings) > max_subspace:
    idx = np.random.choice(len(bitstrings), max_subspace, replace=False)
    bitmatrix = bitmatrix[idx]
print(f"Collected {bitmatrix.shape[0]} bitstrings for SQD.")

# ============================================================
# SQD to find approximate optimal portfolios
# ============================================================

energies, states = solve_qubit(bitmatrix, H, k=max_eigvals)
print("Approximate optimal portfolio energies:")
for i,val in enumerate(energies):
    print(f"{i}: {val:.6f}")

# ============================================================
# Map bitstrings to portfolio allocations
# ============================================================

best_idx = np.argmin(energies)
best_portfolio = bitmatrix[best_idx]
print("Best portfolio selection (1=buy,0=skip):")
for i, sel in enumerate(best_portfolio):
    print(f"Asset {i}: {'BUY' if sel else 'SKIP'}")

# ============================================================
# Plot approximate energies
# ============================================================

plt.figure(figsize=(6,4))
plt.plot(range(len(energies)), energies, 'o-')
plt.xlabel("Eigenvalue index")
plt.ylabel("Portfolio Hamiltonian energy")
plt.title("SQD Approximate Portfolio Optimization")
plt.grid(True)
plt.tight_layout()
plt.savefig("portfolio_sqd_energies.png")
print("Plot saved to portfolio_sqd_energies.png")

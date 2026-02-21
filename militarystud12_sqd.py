import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit_aer import AerSimulator

# SQD function for subspace diagonalization
from qiskit_addon_sqd.qubit import solve_qubit

# ============================================================
# Parameters
# ============================================================

n = 16        # number of qubits
J = 1.2       # XX/YY coupling
Delta = 0.8   # ZZ coupling
B = 0.3       # local Z field
t = 1.0       # evolution time
shots = 2000  # number of samples

# ============================================================
# Build XXZ Hamiltonian
# ============================================================

def build_hamiltonian(n, J, Delta, B):
    pauli_list = []

    def term(pauli, i, j=None, coeff=1.0):
        label = ["I"] * n
        label[i] = pauli[0]
        if j is not None:
            label[j] = pauli[1]
        return ("".join(label), coeff)

    for i in range(n - 1):
        pauli_list.append(term("XX", i, i + 1, J))
        pauli_list.append(term("YY", i, i + 1, J))
        pauli_list.append(term("ZZ", i, i + 1, J * Delta))

    for i in range(n):
        label = ["I"] * n
        label[i] = "Z"
        pauli_list.append(("".join(label), B))

    return SparsePauliOp.from_list(pauli_list).simplify()


H = build_hamiltonian(n, J, Delta, B)

# ============================================================
# Build Ansatz Circuit
# ============================================================

qc = QuantumCircuit(n)

# Use PauliEvolutionGate with Lie-Trotter synthesis
evolution_gate = PauliEvolutionGate(H, time=t, synthesis=LieTrotter(reps=1))
qc.append(evolution_gate, range(n))

# Optional decomposition for stability
qc = qc.decompose(reps=5)

# ============================================================
# Backend
# ============================================================

backend = AerSimulator()  # default supports shot-based counts
qc = transpile(qc, backend, optimization_level=1)

# ============================================================
# Run Circuit and Collect Bitstrings
# ============================================================

print("Sampling bitstrings from evolution circuit...")
job = backend.run(qc, shots=shots)
result = job.result()
counts = result.get_counts()

bitstrings = list(counts.keys())
bitmatrix = np.array([[int(b) for b in bs] for bs in bitstrings], dtype=bool)

print(f"Collected {len(bitstrings)} unique bitstrings for SQD.")

# ============================================================
# Diagonalize Hamiltonian in Sampled Subspace
# ============================================================

energies, states = solve_qubit(bitmatrix, H)

print("Approximate eigenvalues (first 10):")
for i, val in enumerate(energies[:10]):
    print(f"{i}: {val:.6f}")

# ============================================================
# Optional: Plot Eigenvalues
# ============================================================

plt.figure(figsize=(8,4))
plt.plot(range(len(energies)), energies, 'o-', markersize=5)
plt.xlabel("Eigenvalue index")
plt.ylabel("Energy (approx.)")
plt.title("SQD Approximate Eigenvalues of 16-Qubit XXZ Hamiltonian")
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit_aer import AerSimulator

# SQD addon
from qiskit_addon_sqd import SQD

# ============================================================
# Parameters
# ============================================================

n = 16        # number of qubits
J = 1.2       # XX/YY coupling
Delta = 0.8   # ZZ coupling
B = 0.3       # local Z field
t = 1.0       # evolution time
shots = 2000  # number of samples per circuit

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
# Backend
# ============================================================

backend = AerSimulator(method="statevector")

# ============================================================
# Build Ansatz Circuit
# ============================================================

qc = QuantumCircuit(n)

# Use PauliEvolutionGate with Trotterization
evolution_gate = PauliEvolutionGate(H, time=t, synthesis=LieTrotter(reps=1))
qc.append(evolution_gate, range(n))

# Transpile for Aer
qc = qc.decompose(reps=5)
qc = transpile(qc, backend, optimization_level=1)

# ============================================================
# Sample-Based Quantum Diagonalization (SQD)
# ============================================================

print("Running SQD on 16-qubit XXZ Hamiltonian...")

sqd = SQD(backend=backend, shots=shots)
eigenvalues, eigenvectors = sqd.run(qc, H)

print("Approximate eigenvalues from SQD:")
for i, val in enumerate(eigenvalues[:10]):  # print first 10
    print(f"{i}: {val:.6f}")

# ============================================================
# Optional: Plot Eigenvalues
# ============================================================

plt.figure(figsize=(8,4))
plt.plot(range(len(eigenvalues)), eigenvalues, 'o-', markersize=5)
plt.xlabel("Eigenvalue index")
plt.ylabel("Energy (approx.)")
plt.title("SQD Approximate Eigenvalues of 16-Qubit XXZ Hamiltonian")
plt.grid(True)
plt.show()

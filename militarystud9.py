# ============================================================
# Imports
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit_aer import AerSimulator


# ============================================================
# Parameters
# ============================================================

n = 16
J = 1.2
Delta = 0.8
B = 0.3
t = 1.0


# ============================================================
# Hamiltonian Builder
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


# ============================================================
# Build Hamiltonian
# ============================================================

H = build_hamiltonian(n, J, Delta, B)

backend = AerSimulator(method="statevector")

print("Computing full correlation matrix...")


# ============================================================
# Build Circuit
# ============================================================

qc = QuantumCircuit(n)

evolution_gate = PauliEvolutionGate(
    H,
    time=t,
    synthesis=LieTrotter(reps=1)
)

qc.append(evolution_gate, range(n))
qc.save_statevector()

qc = qc.decompose(reps=5)
qc = transpile(qc, backend)

result = backend.run(qc).result()
state = Statevector(result.get_statevector())


# ============================================================
# Full ZZ Correlation Matrix
# ============================================================

correlation = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        label = ["I"] * n
        label[i] = "Z"
        label[j] = "Z"
        op = SparsePauliOp.from_list([("".join(label), 1.0)])
        correlation[i, j] = state.expectation_value(op).real


plt.imshow(correlation)
plt.colorbar()
plt.title("Full ZZ Correlation Matrix")
plt.show()

print("Done.")

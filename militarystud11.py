import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit_aer import AerSimulator

from qiskit_addon_sqd.counts import generate_counts_bipartite_hamming
from qiskit_addon_sqd.qubit import solve_qubit

# Build XXZ Hamiltonian (same as before)
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

n = 16
J = 1.2
Delta = 0.8
B = 0.3
t = 1.0
shots = 2000

H = build_hamiltonian(n, J, Delta, B)

backend = AerSimulator()

# Build evolution circuit
qc = QuantumCircuit(n)
evolution_gate = PauliEvolutionGate(H, time=t, synthesis=LieTrotter(reps=1))
qc.append(evolution_gate, range(n))

qc = qc.decompose(reps=5)
qc = transpile(qc, backend)

print("Sampling bitstrings...")
job = backend.run(qc, shots=shots)
result = job.result()
counts = result.get_counts()

# Convert counts into a bitstring matrix
bitstrings = list(counts.keys())
bitmatrix = np.array([[int(b) for b in bs] for bs in bitstrings], dtype=bool)

print("Diagonalizing in bitstring subspace...")
energies, states = solve_qubit(bitmatrix, H)

print("Approximate eigenvalues:")
print(energies)

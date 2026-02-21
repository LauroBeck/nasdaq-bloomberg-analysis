import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate

# -------------------------------
# Parameters
# -------------------------------

n = 16
J = 1.2
Delta = 0.8
B = 0.3
t = 0.5

pauli_list = []

def term(pauli, i, j=None, coeff=1.0):
    label = ["I"] * n
    label[i] = pauli[0]
    if j is not None:
        label[j] = pauli[1]
    return ("".join(label), coeff)

# -------------------------------
# Build Hamiltonian FIRST
# -------------------------------

for i in range(n - 1):
    pauli_list.append(term("XX", i, i+1, J))
    pauli_list.append(term("YY", i, i+1, J))
    pauli_list.append(term("ZZ", i, i+1, J))
    pauli_list.append(term("ZZ", i, i+1, Delta))

for i in range(n):
    label = ["I"] * n
    label[i] = "Z"
    pauli_list.append(("".join(label), B))

hamiltonian = SparsePauliOp.from_list(pauli_list)

print("Hamiltonian built with", len(pauli_list), "terms")

# -------------------------------
# Evolution Gate
# -------------------------------

evo_gate = PauliEvolutionGate(hamiltonian, time=t)

qc = QuantumCircuit(n)
qc.append(evo_gate, range(n))

# -------------------------------
# Draw Circuit
# -------------------------------

print(qc.draw("text"))

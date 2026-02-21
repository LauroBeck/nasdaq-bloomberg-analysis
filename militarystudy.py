import numpy as np
from qiskit.quantum_info import SparsePauliOp

# -------------------------------
# Parameters (material physics)
# -------------------------------

n = 16          # 16-spin lattice
J = 1.2         # exchange coupling
Delta = 0.8     # anisotropy strength
B = 0.3         # external field

pauli_list = []

# -------------------------------
# Helper: build Pauli string
# -------------------------------

def term(pauli, i, j=None, coeff=1.0):
    label = ["I"] * n
    label[i] = pauli[0]
    if j is not None:
        label[j] = pauli[1]
    return ("".join(label), coeff)

# -------------------------------
# Exchange interactions
# -------------------------------

for i in range(n - 1):

    pauli_list.append(term("XX", i, i+1, J))
    pauli_list.append(term("YY", i, i+1, J))
    pauli_list.append(term("ZZ", i, i+1, J))

# -------------------------------
# Anisotropy (rare-earth effect)
# -------------------------------

for i in range(n - 1):
    pauli_list.append(term("ZZ", i, i+1, Delta))

# -------------------------------
# External field
# -------------------------------

for i in range(n):
    label = ["I"] * n
    label[i] = "Z"
    pauli_list.append(("".join(label), B))

# -------------------------------
# Hamiltonian
# -------------------------------

hamiltonian = SparsePauliOp.from_list(pauli_list)

print("Hamiltonian built with", len(pauli_list), "terms")

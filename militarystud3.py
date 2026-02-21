from qiskit.quantum_info import SparsePauliOp

def build_hamiltonian(n, J, Delta, B):

    pauli_list = []

    def term(pauli, i, j=None, coeff=1.0):
        label = ["I"] * n
        label[i] = pauli[0]
        if j is not None:
            label[j] = pauli[1]
        return ("".join(label), coeff)

    # Exchange + anisotropy
    for i in range(n - 1):
        pauli_list.append(term("XX", i, i+1, J))
        pauli_list.append(term("YY", i, i+1, J))
        pauli_list.append(term("ZZ", i, i+1, J))
        pauli_list.append(term("ZZ", i, i+1, Delta))

    # Field
    for i in range(n):
        label = ["I"] * n
        label[i] = "Z"
        pauli_list.append(("".join(label), B))

    return SparsePauliOp.from_list(pauli_list)

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_aer import AerSimulator


# ============================================================
# Hamiltonian Builder (Compact)
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
# Parameters
# ============================================================

n = 16
J = 1.2
Delta = 0.8
B = 0.3

times = np.linspace(0, 2.0, 10)   # EVEN FEWER → MUCH SAFER

H = build_hamiltonian(n, J, Delta, B)

backend = AerSimulator(method="statevector")

energy_vs_time = []

print("Running stable evolution...")

for t in times:

    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(H, time=t), range(n))
    qc.save_statevector()

    result = backend.run(qc).result()
    state = result.get_statevector()

    # ✅ NO DENSE MATRIX — SAFE
    energy = Statevector(state).expectation_value(H).real

    energy_vs_time.append(energy)


plt.figure()
plt.plot(times, energy_vs_time)
plt.xlabel("Time")
plt.ylabel("Energy Expectation ⟨H⟩")
plt.title("Stable 16-Qubit Evolution (Matrix-Free)")
plt.show()

print("Simulation complete.")

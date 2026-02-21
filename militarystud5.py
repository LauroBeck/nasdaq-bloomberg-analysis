import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_aer import AerSimulator


# ============================================================
# Hamiltonian Builder (Reduced & Stable)
# ============================================================

def build_hamiltonian(n, J, Delta, B):

    pauli_list = []

    def term(pauli, i, j=None, coeff=1.0):
        label = ["I"] * n
        label[i] = pauli[0]
        if j is not None:
            label[j] = pauli[1]
        return ("".join(label), coeff)

    # Heisenberg-like interaction
    for i in range(n - 1):
        pauli_list.append(term("XX", i, i + 1, J))
        pauli_list.append(term("YY", i, i + 1, J))
        pauli_list.append(term("ZZ", i, i + 1, J * Delta))

    # External magnetic field
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

times = np.linspace(0, 2.0, 15)   # FEWER STEPS → MUCH STABLER

print("Building Hamiltonian...")
H = build_hamiltonian(n, J, Delta, B)

print("Hamiltonian terms:", len(H))


# ============================================================
# Aer Backend
# ============================================================

backend = AerSimulator(method="statevector")


# ============================================================
# Time Evolution
# ============================================================

energy_vs_time = []

print("Running Aer evolution...")

for t in times:

    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(H, time=t), range(n))

    qc.save_statevector()

    result = backend.run(qc).result()
    state = result.get_statevector()

    energy = np.real(np.vdot(state, H.to_matrix() @ state))

    energy_vs_time.append(energy)


# ============================================================
# Plot
# ============================================================

plt.figure()
plt.plot(times, energy_vs_time)
plt.xlabel("Time")
plt.ylabel("Energy Expectation ⟨H⟩")
plt.title("Stable Aer Time Evolution (16 Qubits)")
plt.show()

print("Simulation complete.")

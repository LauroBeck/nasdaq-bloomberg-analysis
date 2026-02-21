import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, partial_trace, entropy
from qiskit.circuit.library import PauliEvolutionGate


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

    # Exchange + anisotropy
    for i in range(n - 1):
        pauli_list.append(term("XX", i, i + 1, J))
        pauli_list.append(term("YY", i, i + 1, J))
        pauli_list.append(term("ZZ", i, i + 1, J))
        pauli_list.append(term("ZZ", i, i + 1, Delta))

    # External field
    for i in range(n):
        label = ["I"] * n
        label[i] = "Z"
        pauli_list.append(("".join(label), B))

    return SparsePauliOp.from_list(pauli_list)


# ============================================================
# Simulation Parameters
# ============================================================

n = 16
B = 0.3

J_vals = np.linspace(0.5, 2.0, 12)
Delta_vals = np.linspace(0.1, 1.5, 12)

times = np.linspace(0, 2.0, 40)


# ============================================================
# 1. Parameter Sweep → Energy Landscape
# ============================================================

print("Running parameter sweep...")

energies = np.zeros((len(J_vals), len(Delta_vals)))

for i, J in enumerate(J_vals):
    for j, Delta in enumerate(Delta_vals):

        H = build_hamiltonian(n, J, Delta, B)

        qc = QuantumCircuit(n)
        qc.append(PauliEvolutionGate(H, time=0.5), range(n))

        state = Statevector.from_instruction(qc)

        energies[i, j] = state.expectation_value(H).real

print("Sweep complete.")


plt.figure()
plt.imshow(energies, origin="lower", aspect="auto")
plt.colorbar(label="Energy Expectation ⟨H⟩")
plt.xlabel("Δ index")
plt.ylabel("J index")
plt.title("Phase-Like Energy Landscape")
plt.show()


# ============================================================
# Choose Representative Material Point
# ============================================================

J = 1.2
Delta = 0.8

print(f"\nUsing material parameters: J={J}, Δ={Delta}, B={B}")

H = build_hamiltonian(n, J, Delta, B)


# ============================================================
# 2. Time Evolution Response
# ============================================================

print("Computing time evolution...")

energy_vs_time = []

for t in times:
    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(H, time=t), range(n))

    state = Statevector.from_instruction(qc)

    energy_vs_time.append(state.expectation_value(H).real)

plt.figure()
plt.plot(times, energy_vs_time)
plt.xlabel("Time")
plt.ylabel("Energy Expectation ⟨H⟩")
plt.title("Time Evolution Response")
plt.show()


# ============================================================
# 3. Spin Correlation Map
# ============================================================

print("Computing spin correlations...")

qc = QuantumCircuit(n)
qc.append(PauliEvolutionGate(H, time=0.5), range(n))
state = Statevector.from_instruction(qc)

correlations = []

for i in range(n - 1):

    label = ["I"] * n
    label[i] = "Z"
    label[i + 1] = "Z"

    op = SparsePauliOp.from_list([("".join(label), 1.0)])

    correlations.append(state.expectation_value(op).real)

plt.figure()
plt.plot(correlations)
plt.xlabel("Lattice Site")
plt.ylabel("⟨ZZ⟩ Correlation")
plt.title("Nearest-Neighbor Spin Correlations")
plt.show()


# ============================================================
# 4. Entanglement Growth
# ============================================================

print("Computing entanglement dynamics...")

entropies = []

for t in times:

    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(H, time=t), range(n))

    state = Statevector.from_instruction(qc)

    reduced = partial_trace(state, list(range(n // 2, n)))

    entropies.append(entropy(reduced))

plt.figure()
plt.plot(times, entropies)
plt.xlabel("Time")
plt.ylabel("Entanglement Entropy")
plt.title("Entanglement Growth Dynamics")
plt.show()


print("\nSimulation complete.")

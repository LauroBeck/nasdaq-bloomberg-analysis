import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


# ============================================================
# Build Correlation Matrix  <Zi Zj>
# ============================================================

print("\nComputing full correlation matrix...")

# Use representative evolved state (choose time)
t_corr = 1.0
n = 16  # number of qubits
qc = QuantumCircuit(n)

evolution_gate = PauliEvolutionGate(
    H,
    time=t_corr,
    synthesis=LieTrotter(reps=1)
)

qc.append(evolution_gate, range(n))
qc.save_statevector()

qc = qc.decompose(reps=5)
qc = transpile(qc, backend, optimization_level=1)

result = backend.run(qc).result()
state = Statevector(result.get_statevector())

corr_matrix = np.zeros((n, n))


def zz_operator(n, i, j):
    label = ["I"] * n
    label[i] = "Z"
    label[j] = "Z"
    return SparsePauliOp.from_list([("".join(label), 1.0)])


for i in range(n):
    for j in range(n):

        op = zz_operator(n, i, j)
        corr_matrix[i, j] = state.expectation_value(op).real


# ============================================================
# Plot Heatmap
# ============================================================

plt.figure()
plt.imshow(corr_matrix, origin="lower", aspect="auto")
plt.colorbar(label="⟨Zi Zj⟩ Correlation")
plt.xlabel("Spin Site j")
plt.ylabel("Spin Site i")
plt.title(f"Spin Correlation Heatmap (t={t_corr})")
plt.show()

print("Correlation analysis complete.")

import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA


# =====================================================
# 1️⃣ Build Market Hamiltonian (SparsePauliOp)
# =====================================================

paulis = [
    "IIII",
    "ZIII",
    "XXII", "YYII",
    "XIXI", "YIYI",
    "XIIX", "YIIY",
    "IZII",
    "IXXI", "IYYI",
    "IXIX", "IYIY",
    "IIZI",
    "IIXX", "IIYY",
    "IIIZ"
]

coeffs = [
    0.0,
    0.5,
    -0.15, -0.15,
    -0.1, -0.1,
    -0.05, -0.05,
    0.6,
    -0.125, -0.125,
    -0.075, -0.075,
    0.4,
    -0.1, -0.1,
    0.7
]

ham = SparsePauliOp(paulis, coeffs)

print("\nHamiltonian (market operator):")
print(ham)


# =====================================================
# 2️⃣ Exact Ground State (Classical Diagonalization)
# =====================================================

matrix = ham.to_matrix()
eigenvalues = np.linalg.eigvalsh(matrix)
exact_ground_energy = np.min(eigenvalues)

print("\nExact ground state energy: %.6f" % exact_ground_energy)


# =====================================================
# 3️⃣ Build Ansatz (Modern API)
# =====================================================

n_qubits = ham.num_qubits
ansatz = efficient_su2(num_qubits=n_qubits, reps=2)


# =====================================================
# 4️⃣ Setup Estimator + Optimizer
# =====================================================

estimator = StatevectorEstimator()  # V2 primitive (compatible with VQE)
optimizer = COBYLA(maxiter=300)


# =====================================================
# 5️⃣ Run VQE
# =====================================================

vqe = VQE(
    estimator=estimator,
    ansatz=ansatz,
    optimizer=optimizer
)

vqe_result = vqe.compute_minimum_eigenvalue(ham)

print("\nVQE ground state energy: %.6f" % vqe_result.eigenvalue.real)


# =====================================================
# 6️⃣ Error Comparison
# =====================================================

error = abs(vqe_result.eigenvalue.real - exact_ground_energy)
print("Absolute error: %.6e" % error)

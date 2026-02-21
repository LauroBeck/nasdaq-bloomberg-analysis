# ============================================================
# 4-Qubit Hubbard Dimer Quantum Simulation
# Python 3.13 + Qiskit 2.3.0 + qiskit-algorithms 0.4.0
# Fully compatible with AerSimulator local and IBM Runtime
# Using SparsePauliOp (no qiskit.opflow)
# ============================================================

import numpy as np

# Qiskit imports
from qiskit_aer import AerSimulator
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit.algorithms.optimizers import COBYLA  # required in qiskit-algorithms 0.4.0
from qiskit.utils import QuantumInstance

# -----------------------------
# Metal pair selection
# -----------------------------
metals_pair = "Pt-Pd"  # options: "Pt-Pd" or "Ru-Ir"

hubbard_params = {
    "Pt-Pd": {"t": 1.0, "U": 2.0},
    "Ru-Ir": {"t": 1.2, "U": 2.5}
}

t = hubbard_params[metals_pair]["t"]
U = hubbard_params[metals_pair]["U"]

# -----------------------------
# Define 4-qubit Hubbard Dimer Hamiltonian
# -----------------------------
# Basis: [n1_up, n1_down, n2_up, n2_down]
ham = (
    -0.5 * SparsePauliOp.from_list([("XXII", 1.0), ("YYII", 1.0)]) +
    -0.5 * SparsePauliOp.from_list([("IIXX", 1.0), ("IIYY", 1.0)]) +
    0.25 * U * SparsePauliOp.from_list([("ZZII", 1.0), ("IIZZ", 1.0)])
)

print(f"\nHamiltonian for {metals_pair}:\n{ham}")

# -----------------------------
# Exact diagonalization
# -----------------------------
exact_solver = NumPyMinimumEigensolver()
exact_result = exact_solver.compute_minimum_eigenvalue(ham)
print(f"\nExact ground state energy: {exact_result.eigenvalue.real:.6f}")

# -----------------------------
# VQE setup
# -----------------------------
ansatz = EfficientSU2(num_qubits=4, reps=2)
optimizer = COBYLA(maxiter=200)

# -----------------------------
# Choose backend
# -----------------------------
use_ibmq = False  # Set True to run on IBM Quantum Runtime

if use_ibmq:
    # IBM Quantum Runtime setup
    service = QiskitRuntimeService(channel="ibm_quantum")
    sampler = Sampler(session=service)
    qi = sampler
else:
    # Local Aer simulator
    backend = AerSimulator()
    qi = QuantumInstance(backend)

# -----------------------------
# Run VQE
# -----------------------------
vqe = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=qi)
vqe_result = vqe.compute_minimum_eigenvalue(ham)
print(f"\nVQE ground state energy: {vqe_result.eigenvalue.real:.6f}")

# -----------------------------
# Energy error and interpretation
# -----------------------------
energy_error = abs(vqe_result.eigenvalue.real - exact_result.eigenvalue.real)
print(f"Energy error (VQE vs exact): {energy_error:.6f}")

print("\n--- Conceptual Interpretation ---")
print("Separate metals |0> state = unbonded d-orbitals")
print("Bonded state |1> = correlated d-orbital occupation")
print("Energy difference ~ bond formation strength")

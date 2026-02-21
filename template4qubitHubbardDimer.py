# ============================================================
# 4-Qubit Hubbard Dimer Quantum Simulation (Pt-Pd or Ru-Ir)
# ============================================================

# -----------------------------
# Imports
# -----------------------------
import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options

# -----------------------------
# Project Parameters
# -----------------------------
metals_pair = "Pt-Pd"  # Options: "Pt-Pd" or "Ru-Ir"

# Hubbard Dimer parameters (t: hopping, U: on-site Coulomb)
hubbard_params = {
    "Pt-Pd": {"t": 1.0, "U": 2.0},
    "Ru-Ir": {"t": 1.2, "U": 2.5}
}

t = hubbard_params[metals_pair]["t"]
U = hubbard_params[metals_pair]["U"]

# -----------------------------
# Define 4-qubit Hubbard Dimer Hamiltonian
# -----------------------------
# Basis ordering: [n1_up, n1_down, n2_up, n2_down]
# Using Jordan-Wigner mapping for fermions -> qubits
# Hamiltonian: H = -t(c1†c2 + c2†c1) + U(n1_up n1_down + n2_up n2_down)

# Pauli representation manually mapped
# X = σ_x, Y = σ_y, Z = σ_z, I = identity
# For 4 qubits, PauliSumOp is used
# This example uses a standard Hubbard Dimer Pauli mapping

hamiltonian = (
    -0.5 * t * (PauliSumOp.from_list([("XXII", 1.0), ("YYII", 1.0)])) +
    -0.5 * t * (PauliSumOp.from_list([("I IXX", 1.0), ("IIYY", 1.0)])) +
    0.25 * U * (PauliSumOp.from_list([("ZZII", 1.0), ("IIZZ", 1.0)]))
)

print(f"Hamiltonian for {metals_pair}:\n{hamiltonian}")

# -----------------------------
# Exact Diagonalization (fallback)
# -----------------------------
exact_solver = NumPyMinimumEigensolver()
exact_result = exact_solver.compute_minimum_eigenvalue(hamiltonian)
print(f"Exact ground state energy: {exact_result.eigenvalue.real:.4f}")

# -----------------------------
# VQE Setup
# -----------------------------
ansatz = EfficientSU2(num_qubits=4, reps=2)
optimizer = None  # Default COBYLA

# Choose backend: local simulator or IBM Quantum Runtime
use_ibmq = False  # Change to True to use IBM QPU/Runtime

if use_ibmq:
    # --------------------------------------------------
    # IBM Quantum Runtime setup
    # --------------------------------------------------
    service = QiskitRuntimeService(channel="ibm_quantum")  # Make sure API token is set
    sampler = Sampler(session=service)
    backend = sampler
    quantum_instance = backend
else:
    # Local simulator
    backend = Aer.get_backend("aer_simulator_statevector")
    quantum_instance = QuantumInstance(backend)

vqe = VQE(
    ansatz=ansatz,
    optimizer=None,  # COBYLA default
    quantum_instance=quantum_instance
)

# -----------------------------
# Run VQE
# -----------------------------
vqe_result = vqe.compute_minimum_eigenvalue(hamiltonian)
print(f"VQE ground state energy: {vqe_result.eigenvalue.real:.4f}")

# -----------------------------
# Interpret results
# -----------------------------
print("\n--- Interpretation ---")
print(f"Separate metals |0> state corresponds to no d-orbital bonding.")
print(f"Bonded state |1> corresponds to correlated d-orbital occupation across sites.")
print(f"Energy difference ~ bond formation strength: {abs(vqe_result.eigenvalue.real - exact_result.eigenvalue.real):.4f}")

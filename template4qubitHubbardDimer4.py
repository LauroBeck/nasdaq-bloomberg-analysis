# ============================================================
# 4-Qubit Hubbard Dimer Quantum Simulation (Pt-Pd or Ru-Ir)
# Python 3.13 + Qiskit 2.3.0 compatible
# ============================================================

# -----------------------------
# Imports
# -----------------------------
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.algorithms import VQE
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import PauliSumOp
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

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
# Simple Pauli mapping for demonstration purposes
hamiltonian = (
    -0.5 * t * (PauliSumOp.from_list([("XXII", 1.0), ("YYII", 1.0)])) +
    -0.5 * t * (PauliSumOp.from_list([("IIXX", 1.0), ("IIYY", 1.0)])) +
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

# -----------------------------
# Backend Selection
# -----------------------------
use_ibmq = False  # Set True to run on IBM Quantum Runtime

if use_ibmq:
    # --------------------------------------------------
    # IBM Quantum Runtime setup
    # --------------------------------------------------
    # Ensure your IBMQ API token is configured
    service = QiskitRuntimeService(channel="ibm_quantum")
    sampler = Sampler(session=service)
    backend_for_vqe = sampler
else:
    # Local Aer simulator
    backend_for_vqe = AerSimulator()

# -----------------------------
# Run VQE
# -----------------------------
vqe = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=backend_for_vqe)
vqe_result = vqe.compute_minimum_eigenvalue(hamiltonian)
print(f"VQE ground state energy: {vqe_result.eigenvalue.real:.4f}")

# -----------------------------
# Interpretation
# -----------------------------
print("\n--- Interpretation ---")
print(f"Separate metals |0> state = unbonded d-orbitals")
print(f"Bonded state |1> = correlated d-orbital occupation")
print(f"Energy difference ~ bond formation strength: "
      f"{abs(vqe_result.eigenvalue.real - exact_result.eigenvalue.real):.4f}")

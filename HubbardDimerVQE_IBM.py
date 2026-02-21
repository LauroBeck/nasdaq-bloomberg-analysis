# ============================================================
# Hubbard Dimer VQE on IBM Quantum
# Python 3.13 compatible
# ============================================================

import numpy as np
from qiskit import IBMQ
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliSumOp
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA

# ============================================================
# Step 1: Load IBM Quantum account
# ============================================================
# Make sure you already saved your IBMQ account token once:
# IBMQ.save_account('YOUR_API_TOKEN')

provider = IBMQ.load_account()
print("Available IBMQ backends:")
for backend in provider.backends():
    print(backend)

# Choose backend: simulator or real device
backend = provider.get_backend('ibmq_qasm_simulator')  # local simulator on IBM cloud
# backend = provider.get_backend('ibmq_lima')          # small real QPU

# ============================================================
# Step 2: Define Hubbard Dimer Hamiltonian (4 qubits)
# H = -t (c1† c2 + c2† c1) + U (n1_up n1_down + n2_up n2_down)
# ============================================================

t = 1.0  # Hopping term
U = 4.0  # On-site Coulomb repulsion

hamiltonian = PauliSumOp.from_list([
    ("XXII", -t), ("YYII", -t),
    ("IIXX", -t), ("IIYY", -t),
    ("ZIZI", U/4), ("IZIZ", U/4),
    ("ZZII", U/4), ("IIZZ", U/4),
])

print("\nHubbard Dimer Hamiltonian:")
print(hamiltonian)

# ============================================================
# Step 3: Compute exact ground state energy
# ============================================================

exact_solver = NumPyMinimumEigensolver()
exact_result = exact_solver.compute_minimum_eigenvalue(hamiltonian)
print("\nExact Ground State Energy:", exact_result.eigenvalue.real)

# ============================================================
# Step 4: Run VQE simulation on IBM Quantum simulator
# ============================================================

ansatz = EfficientSU2(num_qubits=4, reps=2, entanglement='full')
optimizer = COBYLA(maxiter=200)

quantum_instance = QuantumInstance(backend, shots=1024)

vqe = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=quantum_instance)
vqe_result = vqe.compute_minimum_eigenvalue(hamiltonian)

print("\nVQE Ground State Energy (IBMQ simulator):", vqe_result.eigenvalue.real)

# ============================================================
# Step 5: Optional: Run on real IBM QPU
# WARNING: uses your IBM Quantum credits
# ============================================================

# backend_qpu = provider.get_backend('ibmq_lima')
# quantum_instance_qpu = QuantumInstance(backend_qpu, shots=8192)
# vqe_qpu = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=quantum_instance_qpu)
# vqe_qpu_result = vqe_qpu.compute_minimum_eigenvalue(hamiltonian)
# print("\nVQE Ground State Energy (IBM QPU):", vqe_qpu_result.eigenvalue.real)

# ============================================================
# Step 6: Interpretation (optional)
# ============================================================

print("\nInterpretation:")
print("Probability |0> ~ metals separate")
print("Probability |1> ~ metals bonded")

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator


# ================================================================
# MATERIAL PARAMETERS (Example: Nd-Fe-B proxy)
# ================================================================

J = 1.5
anisotropy = 3.0
field = 0.0
theta = np.pi / 3


# ================================================================
# HAMILTONIAN (Interaction Physics)
# ================================================================

H = SparsePauliOp.from_list([
    ("XX", J),
    ("YY", J),
    ("ZZ", J),
    ("ZZ", anisotropy),
    ("ZI", field),
    ("IZ", field),
])


# ================================================================
# CIRCUIT (Spin Experiment)
# ================================================================

qc = QuantumCircuit(2)

# --- State preparation (excited configuration)
qc.h(0)
qc.h(1)

# --- Spin rotations (control knobs)
qc.ry(theta, 0)
qc.ry(-theta, 1)

# --- Coupling interaction (exchange channel)
qc.cz(0, 1)


# ================================================================
# DRAW CIRCUIT (critical for interpretation)
# ================================================================

print("\nQuantum Spin Interaction Circuit:\n")
print(qc.draw())


# ================================================================
# ENERGY MEASUREMENT
# ================================================================

estimator = StatevectorEstimator()
job = estimator.run([(qc, H)])
result = job.result()

energy = result[0].data.evs

print("\nRotation θ:", theta)
print("Measured Energy:", energy)

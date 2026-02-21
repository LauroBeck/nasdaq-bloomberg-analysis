import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler

from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms.classifiers import QSVC

# --------------------------------------------------
# 1. Synthetic Spin Dataset (J, B, Δ)
# --------------------------------------------------

X_train = np.array([
    [0.5, 1.2, 0.1],   # weak coupling → Phase 0
    [0.6, 1.0, 0.2],
    [1.8, 0.2, 2.5],   # strong coupling → Phase 1
    [1.5, 0.3, 2.0],
])

y_train = np.array([0, 0, 1, 1])

# --------------------------------------------------
# 2. Spin-Inspired Feature Map
# --------------------------------------------------

params = ParameterVector("θ", 3)

qc = QuantumCircuit(2)

# Encode physical parameters as rotations
qc.ry(params[0], 0)        # J → spin rotation
qc.rz(params[1], 1)        # B → field influence
qc.cx(0, 1)                # interaction / correlation
qc.ry(params[2], 1)        # Δ → anisotropy effect

feature_map = qc

# --------------------------------------------------
# 3. Kernel via State Fidelity
# --------------------------------------------------

sampler = StatevectorSampler()
fidelity = ComputeUncompute(sampler=sampler)

kernel = FidelityQuantumKernel(
    feature_map=feature_map,
    fidelity=fidelity
)

# --------------------------------------------------
# 4. Quantum Classifier
# --------------------------------------------------

classifier = QSVC(quantum_kernel=kernel)

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_train)

print("\nPredictions:", predictions)

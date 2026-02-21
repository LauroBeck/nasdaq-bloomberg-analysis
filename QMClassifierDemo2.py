import numpy as np

from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms.classifiers import QSVC

# -----------------------------
# Dataset
# -----------------------------
X_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_train = np.array([0, 1, 1, 0])

# -----------------------------
# Feature Map
# -----------------------------
feature_map = ZZFeatureMap(feature_dimension=2, reps=2)

# -----------------------------
# Fidelity Primitive (NEW PART)
# -----------------------------
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)

# -----------------------------
# Kernel
# -----------------------------
kernel = FidelityQuantumKernel(
    feature_map=feature_map,
    fidelity=fidelity
)

# -----------------------------
# Classifier
# -----------------------------
classifier = QSVC(quantum_kernel=kernel)

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_train)

print("Predictions:", predictions)

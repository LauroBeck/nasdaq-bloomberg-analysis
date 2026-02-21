import numpy as np

from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import AerSimulator

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms.classifiers import QSVC

# -----------------------------
# Dataset (simple demo)
# -----------------------------
X_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_train = np.array([0, 1, 1, 0])

# -----------------------------
# Quantum Feature Map
# -----------------------------
feature_map = ZZFeatureMap(feature_dimension=2, reps=2)

# -----------------------------
# Backend / Simulator
# -----------------------------
backend = AerSimulator()

# -----------------------------
# Kernel (modern replacement)
# -----------------------------
kernel = FidelityQuantumKernel(
    feature_map=feature_map,
    backend=backend
)

# -----------------------------
# Classifier
# -----------------------------
classifier = QSVC(quantum_kernel=kernel)

classifier.fit(X_train, y_train)

# -----------------------------
# Prediction test
# -----------------------------
predictions = classifier.predict(X_train)

print("Predictions:", predictions)

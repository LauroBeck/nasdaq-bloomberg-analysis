# QMClassifierDemo.py

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.utils import derive_kernel
from qiskit.algorithms.optimizers import COBYLA

# Example dataset: energy signatures labelled by material class
# Here we generate simple synthetic features from your spin model sweep.
def generate_spin_features(material_key, steps=20):
    from MultiMaterialHamiltonianModel1 import MATERIALS, build_hamiltonian, build_circuit
    params = MATERIALS[material_key]
    H = build_hamiltonian(**params)
    estimator = StatevectorEstimator()
    thetas = np.linspace(0, np.pi, steps)
    energies = []
    for theta in thetas:
        qc = build_circuit(theta)
        job = estimator.run([(qc, H)])
        energies.append(job.result()[0].data.evs)
    return np.array(energies)

# Labels: 0 for one class (e.g., “magnet strong”), 1 for another (e.g., “magnet weak”)
data0 = generate_spin_features("nd_fe_b")
data1 = generate_spin_features("halide_RE")

X = np.vstack([data0, data1])
y = np.array([0]*len(data0) + [1]*len(data1))

# Build Quantum Kernel and VQC
def feature_map(x):
    # Simple state preparation based on energies
    qc = QuantumCircuit(2)
    qc.ry(x[0] % (2*np.pi), 0)
    qc.ry(x[-1] % (2*np.pi), 1)
    return qc

qkernel = QuantumKernel(feature_map=feature_map)
vqc = VQC(
    optimizer=COBYLA(maxiter=100),
    quantum_kernel=qkernel
)

# Train classifier
vqc.fit(X, y)
accuracy = vqc.score(X, y)
print("VQC classification accuracy:", accuracy)

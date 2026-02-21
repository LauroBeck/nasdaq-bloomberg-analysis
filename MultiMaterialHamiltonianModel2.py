import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator


# ================================================================
# MATERIAL LIBRARY
# ================================================================

MATERIALS = {

    "oxide_RE": {"J": 1.0, "anisotropy": 2.0, "field": 0.2},
    "halide_RE": {"J": 0.4, "anisotropy": 0.3, "field": 0.1},
    "cerium_like": {"J": 0.8, "anisotropy": 0.5, "field": 0.6},
    "nd_fe_b": {"J": 1.5, "anisotropy": 3.0, "field": 0.0},
    "sm_co": {"J": 1.8, "anisotropy": 2.5, "field": 0.0},
    "terfenol_like": {"J": 1.2, "anisotropy": 4.0, "field": 0.3},
    "ybco_like": {"J": 0.9, "anisotropy": 0.1, "field": 0.0}
}


# ================================================================
# HAMILTONIAN
# ================================================================

def build_hamiltonian(J, anisotropy, field):

    return SparsePauliOp.from_list([
        ("XX", J),
        ("YY", J),
        ("ZZ", J),
        ("ZZ", anisotropy),
        ("ZI", field),
        ("IZ", field),
    ])


# ================================================================
# CIRCUIT
# ================================================================

def build_circuit(theta):

    qc = QuantumCircuit(2)

    qc.h(0)
    qc.h(1)

    qc.ry(theta, 0)
    qc.ry(-theta, 1)

    qc.cz(0, 1)

    return qc


# ================================================================
# ENERGY
# ================================================================

def energy_for_theta(theta, H, estimator):

    qc = build_circuit(theta)
    job = estimator.run([(qc, H)])
    result = job.result()

    return result[0].data.evs


# ================================================================
# PLOT SINGLE MATERIAL
# ================================================================

def plot_single_material(material_key, steps=50):

    params = MATERIALS[material_key]
    H = build_hamiltonian(**params)

    estimator = StatevectorEstimator()

    thetas = np.linspace(0, np.pi, steps)
    energies = [energy_for_theta(t, H, estimator) for t in thetas]

    plt.figure()
    plt.plot(thetas, energies)
    plt.title(f"Energy Landscape → {material_key}")
    plt.xlabel("Rotation θ")
    plt.ylabel("Energy")
    plt.show()


# ================================================================
# MULTI-MATERIAL COMPARISON
# ================================================================

def compare_materials(material_list, steps=50):

    estimator = StatevectorEstimator()
    thetas = np.linspace(0, np.pi, steps)

    plt.figure()

    for material_key in material_list:

        params = MATERIALS[material_key]
        H = build_hamiltonian(**params)

        energies = [energy_for_theta(t, H, estimator) for t in thetas]

        plt.plot(thetas, energies, label=material_key)

    plt.title("Multi-Material Energy Comparison")
    plt.xlabel("Rotation θ")
    plt.ylabel("Energy")
    plt.legend()
    plt.show()


# ================================================================
# ENTRY
# ================================================================

if __name__ == "__main__":

    plot_single_material("nd_fe_b")

    compare_materials([
        "oxide_RE",
        "halide_RE",
        "nd_fe_b",
        "sm_co",
        "terfenol_like",
        "ybco_like"
    ])

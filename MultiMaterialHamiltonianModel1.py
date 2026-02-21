import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator


# ================================================================
# MATERIAL LIBRARY
# Effective parameters representing rare-earth physics
# ================================================================

MATERIALS = {

    "oxide_RE": {
        "J": 1.0,
        "anisotropy": 2.0,
        "field": 0.2
    },

    "halide_RE": {
        "J": 0.4,
        "anisotropy": 0.3,
        "field": 0.1
    },

    "cerium_like": {
        "J": 0.8,
        "anisotropy": 0.5,
        "field": 0.6
    },

    "nd_fe_b": {
        "J": 1.5,
        "anisotropy": 3.0,
        "field": 0.0
    },

    "sm_co": {
        "J": 1.8,
        "anisotropy": 2.5,
        "field": 0.0
    },

    "terfenol_like": {
        "J": 1.2,
        "anisotropy": 4.0,
        "field": 0.3
    },

    "ybco_like": {
        "J": 0.9,
        "anisotropy": 0.1,
        "field": 0.0
    }
}


# ================================================================
# HAMILTONIAN BUILDER
# ================================================================

def build_hamiltonian(J, anisotropy, field):
    """
    Two-spin effective Hamiltonian:
    Heisenberg exchange + anisotropy + external field
    """

    terms = [

        ("XX", J),
        ("YY", J),
        ("ZZ", J),

        # Rare-earth crystal field proxy
        ("ZZ", anisotropy),

        # External / lattice field
        ("ZI", field),
        ("IZ", field),
    ]

    return SparsePauliOp.from_list(terms)


# ================================================================
# CIRCUIT BUILDER (Spin Manipulation)
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
# ENERGY EVALUATION
# ================================================================

def evaluate_energy(qc, H, estimator):
    job = estimator.run([(qc, H)])
    result = job.result()

    return result[0].data.evs


# ================================================================
# MAIN EXPERIMENT
# ================================================================

def run_single_material(material_key, theta):

    params = MATERIALS[material_key]

    H = build_hamiltonian(
        J=params["J"],
        anisotropy=params["anisotropy"],
        field=params["field"]
    )

    qc = build_circuit(theta)

    estimator = StatevectorEstimator()

    energy = evaluate_energy(qc, H, estimator)

    print("\n==============================")
    print("Material Model:", material_key)
    print("Parameters:", params)
    print("Rotation θ:", theta)
    print("Energy:", energy)


# ================================================================
# SCAN MODE (CERN-style parameter exploration)
# ================================================================

def parameter_scan(material_key, steps=12):

    params = MATERIALS[material_key]

    H = build_hamiltonian(
        J=params["J"],
        anisotropy=params["anisotropy"],
        field=params["field"]
    )

    estimator = StatevectorEstimator()

    print("\n========================================")
    print("SCAN MODE →", material_key)
    print("========================================")

    for theta in np.linspace(0, np.pi, steps):

        qc = build_circuit(theta)
        energy = evaluate_energy(qc, H, estimator)

        print(f"θ={theta:.3f}  Energy={energy:.6f}")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    SELECTED_MATERIAL = "nd_fe_b"     # Change freely
    THETA = np.pi / 3

    run_single_material(SELECTED_MATERIAL, THETA)

    # Optional scan
    parameter_scan(SELECTED_MATERIAL)

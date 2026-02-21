from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import SLSQP


# ------------------------------------------------------------------
# MATERIAL PARAMETER LIBRARY (Effective Spin Physics)
# ------------------------------------------------------------------

materials = {

    # Rare-earth oxide archetype (Ln2O3 behavior proxy)
    "oxide_RE": {
        "J": 1.0,          # exchange
        "anisotropy": 2.0, # strong crystal field effects
        "field": 0.2
    },

    # Cerium IV anomaly proxy (mixed valence / different symmetry)
    "cerium_like": {
        "J": 0.8,
        "anisotropy": 0.5,
        "field": 0.6
    },

    # Halide-like systems (weaker coupling, more ionic)
    "halide_RE": {
        "J": 0.4,
        "anisotropy": 0.3,
        "field": 0.1
    },

    # Nd-Fe-B magnet proxy
    "nd_fe_b": {
        "J": 1.5,
        "anisotropy": 3.0,
        "field": 0.0
    },

    # Sm-Co magnet proxy
    "sm_co": {
        "J": 1.8,
        "anisotropy": 2.5,
        "field": 0.0
    },

    # Terfenol-D proxy (TbDyFe2 magnetostriction-like)
    "terfenol_like": {
        "J": 1.2,
        "anisotropy": 4.0,
        "field": 0.3
    },

    # YBCO-inspired toy correlated system
    "ybco_like": {
        "J": 0.9,
        "anisotropy": 0.1,
        "field": 0.0
    }
}


# ------------------------------------------------------------------
# HAMILTONIAN BUILDER
# ------------------------------------------------------------------

def build_hamiltonian(J, anisotropy, field):

    terms = [

        # Heisenberg exchange
        ("XX", J),
        ("YY", J),
        ("ZZ", J),

        # Easy-axis anisotropy (REE physics proxy)
        ("ZZ", anisotropy),

        # External / crystal field
        ("ZI", field),
        ("IZ", field),
    ]

    return SparsePauliOp.from_list(terms)


# ------------------------------------------------------------------
# SELECT MATERIAL SYSTEM
# ------------------------------------------------------------------

selected_material = "nd_fe_b"   # change freely

params = materials[selected_material]

hamiltonian = build_hamiltonian(
    J=params["J"],
    anisotropy=params["anisotropy"],
    field=params["field"]
)


# ------------------------------------------------------------------
# VARIATIONAL CIRCUIT
# ------------------------------------------------------------------

ansatz = TwoLocal(
    num_qubits=2,
    rotation_blocks="ry",
    entanglement_blocks="cz",
    reps=3
)

optimizer = SLSQP()
estimator = Estimator()

vqe = VQE(estimator, ansatz, optimizer)

result = vqe.compute_minimum_eigenvalue(hamiltonian)

print(f"Material model: {selected_material}")
print("Ground state energy:", result.eigenvalue.real)

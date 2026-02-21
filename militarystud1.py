from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate

# -----------------------------------
# Evolution time (physical parameter)
# -----------------------------------

t = 0.5   # arbitrary simulation time

# -----------------------------------
# Build evolution gate
# -----------------------------------

evo_gate = PauliEvolutionGate(hamiltonian, time=t)

# -----------------------------------
# Create circuit
# -----------------------------------

qc = QuantumCircuit(n)
qc.append(evo_gate, range(n))

# -----------------------------------
# Draw circuit
# -----------------------------------

print(qc.draw("text"))

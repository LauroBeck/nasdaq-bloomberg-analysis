# NasdaqBloomberg2.py
import yfinance as yf
import numpy as np
from itertools import product

from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA

# =====================================================
# 1️⃣ Pull real price data
# =====================================================
tickers = ["IBM", "SPY"]  # IBM + Bloomberg index proxy
prices = {}

for t in tickers:
    data = yf.Ticker(t).history(period="1y")["Close"]
    prices[t] = data.values

price_matrix = np.column_stack([prices[t] for t in tickers])

# =====================================================
# 2️⃣ Compute returns and covariance
# =====================================================
returns = (price_matrix[1:] - price_matrix[:-1]) / price_matrix[:-1]
mu = np.mean(returns, axis=0)
cov = np.cov(returns.T)

print("\nExpected returns:", mu)
print("\nCovariance matrix:\n", cov)

# =====================================================
# 3️⃣ Convert portfolio problem to QUBO → Pauli Hamiltonian
# =====================================================
lambda_risk = 0.5
num_assets = len(tickers)

# QUBO coefficients: H = x^T Q x - lambda*mu^T x
Q = cov
c = -lambda_risk * mu

paulis = []
coeffs = []

# Diagonal terms
for i in range(num_assets):
    z_str = ['I']*num_assets
    z_str[i] = 'Z'
    paulis.append(''.join(z_str))
    coeffs.append(Q[i,i]/2 + c[i]/2)

# Off-diagonal terms
for i in range(num_assets):
    for j in range(i+1, num_assets):
        z_str = ['I']*num_assets
        z_str[i] = 'Z'
        z_str[j] = 'Z'
        paulis.append(''.join(z_str))
        coeffs.append(Q[i,j]/4)

ham = SparsePauliOp(paulis, coeffs)
print("\nHamiltonian (Pauli operator):")
print(ham)

# =====================================================
# 4️⃣ Build Ansatz + Estimator + Optimizer
# =====================================================
ansatz = efficient_su2(num_qubits=num_assets, reps=2)
estimator = StatevectorEstimator()
optimizer = COBYLA(maxiter=300)

vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)

# =====================================================
# 5️⃣ Run VQE
# =====================================================
vqe_result = vqe.compute_minimum_eigenvalue(ham)
print("\nVQE ground state energy:", vqe_result.eigenvalue.real)

# =====================================================
# 6️⃣ Extract optimal portfolio
# =====================================================
sv = Statevector(ansatz.assign_parameters(vqe_result.optimal_point))
probs = sv.probabilities_dict()

# Select the most probable configuration
optimal_portfolio = max(probs, key=probs.get)

print("\nOptimal portfolio bitstring:", optimal_portfolio)
for i, bit in enumerate(optimal_portfolio):
    print(f"{tickers[i]}: {'Invest' if bit=='1' else 'Skip'}")

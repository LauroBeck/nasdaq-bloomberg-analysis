import datetime
import numpy as np
import matplotlib.pyplot as plt
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

# 1. Gerar dados fictícios para 4 ativos financeiros
num_assets = 4
stocks = [f"TICKER{i}" for i in range(num_assets)]
data = RandomDataProvider(tickers=stocks,
                          start=datetime.datetime(2023, 1, 1),
                          end=datetime.datetime(2023, 1, 30))
data.run()

# Obter retornos médios e matriz de covariância (risco)
mu = data.get_period_return_mean_vector()
sigma = data.get_period_return_covariance_matrix()

# 2. Definir o problema de otimização
q = 0.5       # Fator de aversão ao risco
budget = 2    # Quantidade de ativos que queremos na carteira
portfolio = PortfolioOptimization(expected_returns=mu, covariances=sigma, 
                                  risk_factor=q, budget=budget)
quadratic_program = portfolio.to_quadratic_program()

# 3. Resolver usando o algoritmo quântico QAOA
sampler = Sampler()
optimizer = COBYLA(maxiter=100)
qaoa = QAOA(sampler=sampler, optimizer=optimizer)

# Converter o problema em um operador quântico e resolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
quantum_optimizer = MinimumEigenOptimizer(qaoa)
result = quantum_optimizer.solve(quadratic_program)

print(f"Ativos selecionados: {result.x}")
print(f"Valor da função objetivo: {result.fval:.4f}")

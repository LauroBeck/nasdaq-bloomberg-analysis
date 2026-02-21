import datetime
import numpy as np
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler # Padrão V2 para Qiskit 1.x
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# 1. Gerar dados financeiros (Simulação de 4 ativos)
num_assets = 4
stocks = [f"TICKER{i}" for i in range(num_assets)]
data = RandomDataProvider(tickers=stocks,
                          start=datetime.datetime(2023, 1, 1),
                          end=datetime.datetime(2023, 1, 30))
data.run()

mu = data.get_period_return_mean_vector()
sigma = data.get_period_return_covariance_matrix()

# 2. Definir o problema de Otimização de Portfólio
q = 0.5       # Aversão ao risco
budget = 2    # Queremos escolher exatamente 2 ativos
portfolio = PortfolioOptimization(expected_returns=mu, covariances=sigma, 
                                  risk_factor=q, budget=budget)
quadratic_program = portfolio.to_quadratic_program()

# 3. Configurar o Algoritmo Quântico (QAOA)
# Usamos o StatevectorSampler para execução local rápida
sampler = StatevectorSampler()
optimizer = COBYLA(maxiter=100)

# Inicializamos o QAOA com o sampler e o otimizador clássico
qaoa = QAOA(sampler=sampler, optimizer=optimizer)

# 4. Resolver o problema
# O MinimumEigenOptimizer faz a ponte entre o problema financeiro e o algoritmo quântico
quantum_optimizer = MinimumEigenOptimizer(qaoa)
result = quantum_optimizer.solve(quadratic_program)

# 5. Exibir resultados
print("--- Resultado da Otimização ---")
print(f"Ativos selecionados (Binário): {result.x}")
selected_tickers = [stocks[i] for i, val in enumerate(result.x) if val > 0]
print(f"Carteira sugerida: {selected_tickers}")
print(f"Valor da função objetivo (Risco/Retorno): {result.fval:.4f}")

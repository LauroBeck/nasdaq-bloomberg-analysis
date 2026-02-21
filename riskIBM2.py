import datetime
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.sparse import SparseEfficiencyWarning

# Silenciar avisos de performance de matrizes esparsas para uma saída limpa
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from qiskit_finance.data_providers import RandomDataProvider
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# ============================================================
# 1. Dados do Mercado (Simulação de 4 Ativos)
# ============================================================
num_assets = 4
stocks = ["AAPL", "GOOG", "MSFT", "AMZN"] # Nomes ilustrativos
data = RandomDataProvider(tickers=stocks,
                          start=datetime.datetime(2023, 1, 1),
                          end=datetime.datetime(2023, 1, 30))
data.run()

mu = data.get_period_return_mean_vector()
sigma = data.get_period_return_covariance_matrix()

# ============================================================
# 2. Modelagem do Problema Financeiro
# ============================================================
q = 0.5       # Fator de aversão ao risco (0.1 = agressivo, 1.0 = conservador)
budget = 2    # Número exato de ativos que queremos na carteira
portfolio = PortfolioOptimization(expected_returns=mu, covariances=sigma, 
                                  risk_factor=q, budget=budget)
quadratic_program = portfolio.to_quadratic_program()

# ============================================================
# 3. Execução Quântica (QAOA com Sampler V2)
# ============================================================
sampler = StatevectorSampler()
optimizer = COBYLA(maxiter=150) # Aumentado para melhor convergência
qaoa = QAOA(sampler=sampler, optimizer=optimizer)

quantum_optimizer = MinimumEigenOptimizer(qaoa)
result = quantum_optimizer.solve(quadratic_program)

# ============================================================
# 4. Análise e Visualização dos Resultados
# ============================================================
print("\n" + "="*30)
print("RELATÓRIO DE OTIMIZAÇÃO QUÂNTICA")
print("="*30)
selected_indices = [i for i, val in enumerate(result.x) if val > 0]
selected_tickers = [stocks[i] for i in selected_indices]

print(f"Status: {result.status}")
print(f"Ativos Escolhidos: {selected_tickers}")
print(f"Valor da Função Objetivo: {result.fval:.6f}")

# Gráfico de Retorno Esperado com Destaque para Selecionados
colors = ['#2ecc71' if i in selected_indices else '#bdc3c7' for i in range(num_assets)]

plt.figure(figsize=(10, 5))
bars = plt.bar(stocks, mu, color=colors, edgecolor='black', alpha=0.8)
plt.axhline(0, color='black', linewidth=1)
plt.title(f'Seleção Quântica: Carteira com {budget} Ativos (Risco q={q})', fontsize=14)
plt.ylabel('Retorno Médio Esperado', fontsize=12)

# Adiciona etiquetas nos ativos selecionados
for i, bar in enumerate(bars):
    if i in selected_indices:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                 'HOLD', ha='center', va='bottom', fontweight='bold', color='#27ae60')

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

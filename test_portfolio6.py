import yfinance as yf
import pandas as pd

# Mapeamento com o sufixo .SA necessário para o Yahoo Finance
ativos_map = {
    "JPMC34.SA": "JPMorgan Chase & CO. BDR",
    "AMZO34.SA": "Amazon BDR",
    "GSGI34.SA": "Goldman Sachs Group BDR",
    "MSFT34.SA": "Microsoft BDR",
    "BOAC34.SA": "Bank Of America BDR"
}

tickers = list(ativos_map.keys())

# Define a data da imagem (11 de fev, 2026)
data_alvo = "2026-02-11"
data_fim = "2026-02-12"

print(f"--- Buscando cotações para {data_alvo} ---")
# auto_adjust=True garante que o preço de fechamento já considere dividendos/splits
df = yf.download(tickers, start=data_alvo, end=data_fim, auto_adjust=True)

# Verifica se o DataFrame contém dados para evitar o erro 'out-of-bounds'
if df.empty:
    print(f"Erro: Não foram encontrados dados para a data {data_alvo}.")
    print("Dica: Verifique se o mercado estava aberto ou se há conexão com a internet.")
else:
    # Acessa os preços de fechamento
    precos_fechamento = df['Close']
    
    print("\nAtivo | Preço na Imagem (BRL) | Preço Capturado (BRL)")
    print("-" * 55)
    
    valores_imagem = {
        "JPMC34.SA": 161.00,
        "AMZO34.SA": 53.06,
        "GSGI34.SA": 165.54,
        "MSFT34.SA": 87.63,
        "BOAC34.SA": 69.66
    }

    for ticker in tickers:
        try:
            # Pega o primeiro valor disponível na série temporal (índice 0)
            val_capturado = precos_fechamento[ticker].iloc[0]
            print(f"{ticker:9} | {valores_imagem[ticker]:>20.2f} | {val_capturado:>20.2f}")
        except (IndexError, KeyError):
            print(f"{ticker:9} | {valores_imagem[ticker]:>20.2f} | Dados não disponíveis")

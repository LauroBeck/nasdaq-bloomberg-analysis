import yfinance as yf
import pandas as pd

# Mapeamento de tickers (BDRs na B3) conforme a imagem
ativos_map = {
    "JPMC34": "JPMorgan Chase & CO. BDR",
    "AMZO34": "Amazon BDR",
    "GSGI34": "Goldman Sachs Group BDR",
    "MSFT34": "Microsoft BDR",
    "BOAC34": "Bank Of America BDR"
}

tickers = list(ativos_map.keys())

# Define a data da imagem (11 de fev, 2026)
# O yfinance requer o dia seguinte no 'end' para incluir o dia desejado
data_alvo = "2026-02-11"
data_fim = "2026-02-12"

print(f"--- Buscando cotações para {data_alvo} ---")
df = yf.download(tickers, start=data_alvo, end=data_fim)

# Extrai os preços de fechamento (Close)
# Nota: Em BDRs, o 'Adj Close' costuma ser igual ao 'Close' no intraday
precos_fechamento = df['Close'].iloc[0]

# Tabela de comparação com a imagem
print("\nAtivo | Preço na Imagem (BRL) | Preço Capturado (BRL)")
print("-" * 55)
for ticker in tickers:
    val_capturado = precos_fechamento[ticker]
    # Valores extraídos visualmente da sua imagem:
    valores_imagem = {
        "JPMC34": 161.00,
        "AMZO34": 53.06,
        "GSGI34": 165.54,
        "MSFT34": 87.63,
        "BOAC34": 69.66
    }
    print(f"{ticker:6} | {valores_imagem[ticker]:>20.2f} | {val_capturado:>20.2f}")

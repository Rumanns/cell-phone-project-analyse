#ANÁLISE EXPLORATÓRIA DE DADOS - EDA

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\Rumanns\\Desktop\\cell phone project analyse\\data\\Amazon_Cell_Phones.csv")
print(df.columns)

# 1. Agrupar por produto (ou marca, ou modelo)
# Vamos começar agrupando por product_name (nome do produto)
# Calcula métricas agregadas por produto

analise_produtos = df.groupby('brand').agg({
	'Price (Dollar)': ['mean', 'count'],
	'discount_percentage': 'mean',
	'rating_out_of_5': 'mean',
	'price_before_discount': 'mean'
}).round(2)

# Renomear as colunas pra ficar legível
analise_produtos.columns = ['preco_medio', 'quantidade_vendida', 
                            'desconto_medio', 'avaliacao_media', 
                            'preco_original_medio']

# Calcular a receita total
analise_produtos['receita_total'] = analise_produtos['preco_medio'] * analise_produtos['quantidade_vendida']

# Ordenar pra ver os campeões de venda
analise_produtos = analise_produtos.sort_values('quantidade_vendida', ascending=False)

print("🏆 TOP 10 MAIS VENDIDOS:")
print(analise_produtos.head(10))





# Identificar produtos que venderam muito (top 20%) mas com desconto alto (acima da média)
q80_vendas = analise_produtos['quantidade_vendida'].quantile(0.80)
media_desconto = analise_produtos['desconto_medio'].mean()

falsos_positivos = analise_produtos[
    (analise_produtos['quantidade_vendida'] >= q80_vendas) & 
    (analise_produtos['desconto_medio'] > media_desconto)
].sort_values('desconto_medio', ascending=False)

print("\n🔴 FALSOS POSITIVOS: Venderam muito, mas com desconto alto")
print("   (Será que deram tanto desconto que mal valeu a pena?)")
print(falsos_positivos[['quantidade_vendida', 'desconto_medio', 'preco_medio', 'receita_total']])





media_avaliacao = analise_produtos['avaliacao_media'].mean()

falsos_positivos_avaliacao = analise_produtos[
    (analise_produtos['quantidade_vendida'] >= q80_vendas) & 
    (analise_produtos['avaliacao_media'] < media_avaliacao)
].sort_values('avaliacao_media', ascending=True)

print("\n🔴 FALSOS POSITIVOS (versão 2): Venderam muito mas têm avaliação baixa")
print("   (Produto popular, mas cliente não gostou - risco de devolução)")
print(falsos_positivos_avaliacao[['quantidade_vendida', 'avaliacao_media', 'preco_medio']])





q20_vendas = analise_produtos['quantidade_vendida'].quantile(0.20)

falsos_negativos = analise_produtos[
    (analise_produtos['quantidade_vendida'] <= q20_vendas) & 
    (analise_produtos['desconto_medio'] < media_desconto)
].sort_values('desconto_medio', ascending=True)

print("\n🟢 FALSOS NEGATIVOS: Venderam pouco, mas quase sem desconto")
print("   (Talvez seja um produto premium que vale a pena manter)")
print(falsos_negativos[['quantidade_vendida', 'desconto_medio', 'preco_medio', 'avaliacao_media']])





import matplotlib.pyplot as plt

# Gráfico: Quantidade Vendida vs Desconto Médio
plt.figure(figsize=(10, 6))

# Todos os produtos (pontos cinza)
plt.scatter(analise_produtos['quantidade_vendida'], 
            analise_produtos['desconto_medio'], 
            alpha=0.5, c='gray', label='Todos os produtos')

# Destacar falsos positivos (vermelho)
plt.scatter(falsos_positivos['quantidade_vendida'], 
            falsos_positivos['desconto_medio'], 
            color='red', s=100, label='⚠️ Vendeu muito, mas com desconto alto')

# Destacar falsos negativos (verde)
plt.scatter(falsos_negativos['quantidade_vendida'], 
            falsos_negativos['desconto_medio'], 
            color='green', s=100, label='✨ Vendeu pouco, mas com desconto baixo')

plt.xlabel('Quantidade Vendida')
plt.ylabel('Desconto Médio (%)')
plt.title('Onde está o dinheiro? Produtos que vendem com ou sem desconto')
plt.axhline(y=media_desconto, color='blue', linestyle='--', label='Média de desconto')
plt.legend()
plt.show()



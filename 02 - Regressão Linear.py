# REGRESSÃO LINEAR - VÍDEO 3
# Pergunta 1: Ponto de Equilíbrio (Preço vs Vendas)
# Pergunta 2: Ponto de Topo (Desconto Ideal)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ============================================
# CARREGAR E PREPARAR OS DADOS
# ============================================

df = pd.read_csv("C:\\Users\\Rumanns\\Desktop\\cell phone project analyse\\data\\Amazon_Cell_Phones.csv")

# Agrupar por marca
vendas_por_marca = df.groupby('brand').size().reset_index(name='quantidade_vendida')
preco_por_marca = df.groupby('brand')['Price (Dollar)'].mean().reset_index()
desconto_por_marca = df.groupby('brand')['discount_percentage'].mean().reset_index()

# Juntar tudo
analise = pd.merge(vendas_por_marca, preco_por_marca, on='brand')
analise = pd.merge(analise, desconto_por_marca, on='brand')

# Filtrar marcas com pelo menos 15 vendas
analise = analise[analise['quantidade_vendida'] >= 15].sort_values('Price (Dollar)', ascending=False)

print("📊 DADOS AGREGADOS POR MARCA:")
print(analise[['brand', 'quantidade_vendida', 'Price (Dollar)', 'discount_percentage']].head())

# ============================================
# PERGUNTA 1: Ponto de Equilíbrio (Preço vs Vendas)
# ============================================

print("\n" + "="*60)
print("🎯 PERGUNTA 1: Ponto de Equilíbrio")
print("="*60)

X1 = analise[['Price (Dollar)']]
Y1 = analise['quantidade_vendida']

modelo1 = LinearRegression()
modelo1.fit(X1, Y1)

coeficiente = modelo1.coef_[0]

print(f"\n📈 Coeficiente Angular: {coeficiente:.4f}")
print(f"Interpretação: Cada aumento de 1 dólar no preço reduz as vendas em {abs(coeficiente):.2f} unidades em média")

# Gráfico da Pergunta 1
plt.figure(figsize=(14, 9))

plt.scatter(X1, Y1, alpha=0.6, s=100, color='steelblue', edgecolors='black', label='Marcas')
plt.plot(X1, modelo1.predict(X1), color='red', linewidth=2, label='Linha de Regressão')

# Nomes das marcas
for i, marca in enumerate(analise['brand']):
    plt.annotate(marca, (X1.iloc[i, 0], Y1.iloc[i]), fontsize=8, alpha=0.8,
                 rotation=30, ha='left', va='bottom',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="gray"))

plt.xlabel('Preço Médio (Dólar)', fontsize=12)
plt.ylabel('Quantidade Vendida', fontsize=12)
plt.title('Pergunta 1: Ponto de Equilíbrio - Preço vs Vendas', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n✅ RESPOSTA 1: Cada 1 dólar de aumento no preço reduz as vendas em {abs(coeficiente):.2f} unidades.")



# ============================================
# PERGUNTA 2: Ponto de Topo (Desconto Ideal)
# ============================================

print("\n" + "="*60)
print("🎯 PERGUNTA 2: Ponto de Topo (Desconto Ideal)")
print("="*60)

# Criar termo quadrático do desconto
analise['desconto_quadrado'] = analise['discount_percentage'] ** 2

X2 = analise[['discount_percentage', 'desconto_quadrado']]
Y2 = analise['quantidade_vendida']

modelo2 = LinearRegression()
modelo2.fit(X2, Y2)

a = modelo2.coef_[1]  # coeficiente quadrático
b = modelo2.coef_[0]  # coeficiente linear
c = modelo2.intercept_

print(f"\n📈 Coeficientes da regressão quadrática:")
print(f"  Termo quadrático (a): {a:.4f}")
print(f"  Termo linear (b): {b:.4f}")
print(f"  Intercepto (c): {c:.2f}")

# Calcular ponto máximo
if a < 0:
    ponto_maximo = -b / (2 * a)
    print(f"\n🎯 DESCONTO IDEAL: {ponto_maximo:.1f}%")
    print(f"Interpretação: A partir de {ponto_maximo:.1f}% de desconto, o ganho em vendas diminui.")
else:
    ponto_maximo = None
    print(f"\n⚠️ Padrão de U (ponto mínimo) ou dados não seguem curva de sino")

# Gerar curva para visualização
descontos_curva = np.linspace(0, 80, 100).reshape(-1, 1)
descontos_quad_curva = descontos_curva ** 2
X_curva = np.hstack([descontos_curva, descontos_quad_curva])
Y_curva = modelo2.predict(X_curva)

# Gráfico da Pergunta 2
plt.figure(figsize=(14, 9))

plt.scatter(analise['discount_percentage'], analise['quantidade_vendida'], 
            alpha=0.6, s=100, color='steelblue', edgecolors='black', label='Marcas')

plt.plot(descontos_curva, Y_curva, color='red', linewidth=2, label='Curva Quadrática')

if a < 0 and ponto_maximo:
    vendas_topo = (a * ponto_maximo**2) + (b * ponto_maximo) + c
    plt.scatter(ponto_maximo, vendas_topo, color='green', s=200, 
                marker='*', zorder=5, label=f'🎯 Desconto Ideal: {ponto_maximo:.0f}%')
    plt.axvline(x=ponto_maximo, color='green', linestyle='--', alpha=0.5)

# Nomes das marcas
for i, marca in enumerate(analise['brand']):
    plt.annotate(marca, (analise['discount_percentage'].iloc[i], analise['quantidade_vendida'].iloc[i]),
                 fontsize=8, alpha=0.8, rotation=30, ha='left', va='bottom',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="gray"))

plt.xlabel('Desconto Médio (%)', fontsize=12)
plt.ylabel('Quantidade Vendida', fontsize=12)
plt.title('Pergunta 2: Ponto de Topo - Qual o Desconto Ideal?', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

if a < 0 and ponto_maximo:
    print(f"\n✅ RESPOSTA 2: O desconto ideal é {ponto_maximo:.1f}%. Acima disso, você joga dinheiro fora.")
else:
    print(f"\n✅ RESPOSTA 2: Não foi identificado um padrão claro de 'desconto ideal' para este conjunto de dados.")

print("\n" + "="*60)
print("🏁 FIM DAS ANÁLISES DE REGRESSÃO LINEAR (Perguntas 1 e 2)")
print("="*60)



# ============================================
# PERGUNTA 3:
# O QUE REALMENTE FAZ UM CELULAR VENDER?
# ============================================

print("\n" + "="*60)
print("🎯 PERGUNTA 3: O que realmente impacta as vendas?")
print("="*60)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ============================================
# PREPARAR DADOS
# ============================================

# Criar dataframe agregado por marca
analise3 = df.groupby('brand').agg({
    'Price (Dollar)': 'mean',
    'discount_percentage': 'mean',
    'rating_out_of_5': 'mean',
    'RAM (GB)': 'mean',
    'Storage (GB)': 'mean',
    'number_of_ratings': 'sum',
    'ID': 'count'
}).reset_index()

# Renomear quantidade vendida
analise3.rename(columns={'ID': 'quantidade_vendida'}, inplace=True)

# Remover nulos
analise3 = analise3.dropna()

# Filtrar marcas relevantes
analise3 = analise3[analise3['quantidade_vendida'] >= 15]

print("\n📊 DADOS UTILIZADOS:")
print(analise3.head())

# ============================================
# VARIÁVEIS
# ============================================

X3 = analise3[
    [
        'Price (Dollar)',
        'discount_percentage',
        'rating_out_of_5',
        'RAM (GB)',
        'Storage (GB)',
        'number_of_ratings'
    ]
]

Y3 = analise3['quantidade_vendida']

# ============================================
# PADRONIZAÇÃO
# ============================================

scaler = StandardScaler()

X3_scaled = scaler.fit_transform(X3)

# ============================================
# MODELO
# ============================================

modelo3 = LinearRegression()
modelo3.fit(X3_scaled, Y3)

coeficientes = modelo3.coef_

variaveis = X3.columns

# ============================================
# RESULTADOS
# ============================================

print("\n📈 IMPACTO DAS VARIÁVEIS:\n")

for nome, coef in zip(variaveis, coeficientes):

    direcao = "AUMENTA" if coef > 0 else "REDUZ"

    print(
        f"{nome}: "
        f"{direcao} vendas "
        f"(Impacto: {coef:.2f})"
    )

# Score do modelo
r2 = modelo3.score(X3_scaled, Y3)

print(f"\n📊 R² do modelo: {r2:.3f}")

# ============================================
# ORGANIZAR IMPORTÂNCIA
# ============================================

importancia = pd.DataFrame({
    'variavel': variaveis,
    'impacto': coeficientes
})

# Valor absoluto para ranking
importancia['impacto_absoluto'] = importancia['impacto'].abs()

# Ordenar
importancia = importancia.sort_values(
    'impacto_absoluto',
    ascending=False
)

print("\n🏆 RANKING DE IMPACTO:")
print(importancia[['variavel', 'impacto']])

# ============================================
# GRÁFICO
# ============================================

plt.figure(figsize=(12,7))

cores = [
    'green' if x > 0 else 'red'
    for x in importancia['impacto']
]

plt.barh(
    importancia['variavel'],
    importancia['impacto'],
    color=cores
)

plt.axvline(x=0, color='black', linestyle='--')

plt.xlabel('Impacto Padronizado')
plt.ylabel('Variáveis')

plt.title(
    'Pergunta 3: O que mais impacta as vendas?'
)

# Mostrar valores
for i, valor in enumerate(importancia['impacto']):
    plt.text(
        valor,
        i,
        f'{valor:.2f}',
        va='center'
    )

plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ RESPOSTA 3:")
print(
    "Os coeficientes padronizados mostram "
    "quais características mais influenciam "
    "as vendas."
)



# ============================================
# PERGUNTA 4:
# ONDE ESTÁ O PONTO DE VIRADA?
# ============================================

print("\n" + "="*60)
print("🎯 PERGUNTA 5: Onde está o ponto de virada?")
print("="*60)

# ============================================
# PREPARAR DADOS
# ============================================

analise5 = df.groupby('brand').agg({
    'rating_out_of_5': 'mean',
    'ID': 'count'
}).reset_index()

analise5.rename(columns={'ID': 'quantidade_vendida'}, inplace=True)

analise5 = analise5.dropna()

analise5 = analise5[
    analise5['quantidade_vendida'] >= 15
]

# ============================================
# DEFINIR LIMIAR
# ============================================

limiar = 4.0

grupo_baixo = analise5[
    analise5['rating_out_of_5'] < limiar
]

grupo_alto = analise5[
    analise5['rating_out_of_5'] >= limiar
]

# ============================================
# MÉDIAS
# ============================================

media_baixo = grupo_baixo['quantidade_vendida'].mean()

media_alto = grupo_alto['quantidade_vendida'].mean()

print(f"\n📉 Média de vendas abaixo de {limiar}: {media_baixo:.2f}")

print(f"📈 Média de vendas acima de {limiar}: {media_alto:.2f}")

# ============================================
# VISUALIZAÇÃO
# ============================================

plt.figure(figsize=(12,7))

# Grupo abaixo
plt.scatter(
    grupo_baixo['rating_out_of_5'],
    grupo_baixo['quantidade_vendida'],
    color='red',
    s=100,
    alpha=0.7,
    label=f'Abaixo de {limiar}'
)

# Grupo acima
plt.scatter(
    grupo_alto['rating_out_of_5'],
    grupo_alto['quantidade_vendida'],
    color='green',
    s=100,
    alpha=0.7,
    label=f'Acima de {limiar}'
)

# Linha do limiar
plt.axvline(
    x=limiar,
    color='black',
    linestyle='--',
    linewidth=2,
    label='Ponto de Virada'
)

plt.xlabel('Rating Médio')
plt.ylabel('Quantidade Vendida')

plt.title(
    'Pergunta 5: Onde está o ponto de virada?'
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()

# ============================================
# CONCLUSÃO
# ============================================

if media_alto > media_baixo:

    diferenca = media_alto - media_baixo

    print("\n✅ RESPOSTA 5:")

    print(
        f"Produtos acima de {limiar} estrelas "
        f'vendem em média {diferenca:.1f} unidades a mais.'
    )

else:

    print(
        "\n✅ Não foi encontrado um padrão "
        "claro de ponto de virada."
    )
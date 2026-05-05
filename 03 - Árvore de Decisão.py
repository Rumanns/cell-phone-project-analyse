# ============================================
# PERGUNTA 3: Quanto Vale Cada Variável?
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ============================================
# CARREGAR E PREPARAR OS DADOS
# ============================================

df = pd.read_csv("C:\\Users\\Rumanns\\Desktop\\cell phone project analyse\\data\\Amazon_Cell_Phones.csv")


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
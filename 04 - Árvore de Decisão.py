# ============================================
# ÁRVORE DE DECISÃO
# PERGUNTA 1:
# QUAL CAMINHO LEVA AO SUCESSO?
# ============================================

print("\n" + "="*60)
print("🌳 PERGUNTA 1: Qual caminho leva ao sucesso?")
print("="*60)

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# ============================================
# PREPARAR DADOS
# ============================================

df = pd.read_csv("C:\\Users\\Rumanns\\Desktop\\cell phone project analyse\\data\\Amazon_Cell_Phones.csv")
print(df.columns)

analise = df.groupby('brand').agg({
    'Price (Dollar)': 'mean',
    'discount_percentage': 'mean',
    'rating_out_of_5': 'mean',
    'RAM (GB)': 'mean',
    'Storage (GB)': 'mean',
    'ID': 'count'
}).reset_index()

# Renomear coluna
analise.rename(
    columns={'ID': 'quantidade_vendida'},
    inplace=True
)

# Remover nulos
analise = analise.dropna()

# Filtrar marcas relevantes
analise = analise[
    analise['quantidade_vendida'] >= 15
]

# ============================================
# CRIAR CLASSE DE SUCESSO
# ============================================

mediana_vendas = analise['quantidade_vendida'].median()

analise['sucesso'] = (
    analise['quantidade_vendida']
    >= mediana_vendas
).astype(int)

print(f"\n📊 Mediana de vendas: {mediana_vendas}")

print("\n✅ 1 = Sucesso")
print("❌ 0 = Baixas vendas")

# ============================================
# VARIÁVEIS
# ============================================

X = analise[
    [
        'Price (Dollar)',
        'discount_percentage',
        'rating_out_of_5',
        'RAM (GB)',
        'Storage (GB)'
    ]
]

y = analise['sucesso']

# ============================================
# MODELO
# ============================================

modelo = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

modelo.fit(X, y)

# ============================================
# VISUALIZAR ÁRVORE
# ============================================

plt.figure(figsize=(18,10))

plot_tree(
    modelo,
    feature_names=X.columns,
    class_names=['Baixa Venda', 'Sucesso'],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title(
    'Pergunta 1: Qual caminho leva ao sucesso?',
    fontsize=16
)

plt.show()

# ============================================
# IMPORTÂNCIA DAS VARIÁVEIS
# ============================================

importancia = pd.DataFrame({
    'Variável': X.columns,
    'Importância': modelo.feature_importances_
})

importancia = importancia.sort_values(
    'Importância',
    ascending=False
)

print("\n🏆 IMPORTÂNCIA DAS VARIÁVEIS:\n")

print(importancia)

# ============================================
# EXEMPLO DE REGRA
# ============================================

print("\n🧠 A árvore criou regras automaticamente.")
print("Exemplo:")
print("SE rating for alto E desconto for alto")
print("ENTÃO chance de sucesso aumenta.")



# ============================================
# PERGUNTA 2:
# QUEM É PARECIDO COM QUEM?
# ============================================

print("\n" + "="*60)
print("🌳 PERGUNTA 2: Quem é parecido com quem?")
print("="*60)

# ============================================
# PREPARAR DADOS
# ============================================

analise2 = df.groupby('brand').agg({
    'Price (Dollar)': 'mean',
    'discount_percentage': 'mean',
    'rating_out_of_5': 'mean',
    'RAM (GB)': 'mean',
    'Storage (GB)': 'mean',
    'ID': 'count'
}).reset_index()

analise2.rename(
    columns={'ID': 'quantidade_vendida'},
    inplace=True
)

analise2 = analise2.dropna()

analise2 = analise2[
    analise2['quantidade_vendida'] >= 15
]

# ============================================
# CRIAR CLASSE
# ============================================

mediana = analise2['quantidade_vendida'].median()

analise2['sucesso'] = (
    analise2['quantidade_vendida']
    >= mediana
).astype(int)

# ============================================
# VARIÁVEIS
# ============================================

X2 = analise2[
    [
        'Price (Dollar)',
        'discount_percentage',
        'rating_out_of_5',
        'RAM (GB)',
        'Storage (GB)'
    ]
]

y2 = analise2['sucesso']

# ============================================
# TREINAR ÁRVORE
# ============================================

modelo2 = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

modelo2.fit(X2, y2)

# ============================================
# IDENTIFICAR FOLHAS
# ============================================

folhas = modelo2.apply(X2)

analise2['grupo'] = folhas

# ============================================
# MOSTRAR PERFIS
# ============================================

print("\n📊 PERFIS IDENTIFICADOS:\n")

for grupo in analise2['grupo'].unique():

    subset = analise2[
        analise2['grupo'] == grupo
    ]

    print(f"\n🌳 Grupo {grupo}")

    print(
        subset[
            [
                'brand',
                'Price (Dollar)',
                'rating_out_of_5',
                'RAM (GB)',
                'quantidade_vendida'
            ]
        ]
    )

# ============================================
# VISUALIZAÇÃO
# ============================================

plt.figure(figsize=(12,8))

scatter = plt.scatter(
    analise2['Price (Dollar)'],
    analise2['rating_out_of_5'],
    c=analise2['grupo'],
    s=200,
    alpha=0.7
)

# Nome das marcas
for i, marca in enumerate(analise2['brand']):

    plt.annotate(
        marca,
        (
            analise2['Price (Dollar)'].iloc[i],
            analise2['rating_out_of_5'].iloc[i]
        ),
        fontsize=8,
        alpha=0.8
    )

plt.xlabel('Preço Médio')
plt.ylabel('Rating Médio')

plt.title(
    'Pergunta 2: Quem é parecido com quem?'
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()

# ============================================
# CONCLUSÃO
# ============================================

print("\n✅ RESPOSTA 2:")
print(
    "A árvore agrupou marcas com "
    "características parecidas automaticamente."
)



# ============================================
# PERGUNTA 3:
# O QUE NÃO IMPORTA?
# ============================================

print("\n" + "="*60)
print("🌳 PERGUNTA 3: O que a árvore ignorou?")
print("="*60)

# ============================================
# PREPARAR DADOS
# ============================================

analise3 = df.groupby('brand').agg({
    'Price (Dollar)': 'mean',
    'discount_percentage': 'mean',
    'rating_out_of_5': 'mean',
    'RAM (GB)': 'mean',
    'Storage (GB)': 'mean',
    'number_of_ratings': 'sum',
    'ID': 'count'
}).reset_index()

analise3.rename(
    columns={'ID': 'quantidade_vendida'},
    inplace=True
)

analise3 = analise3.dropna()

analise3 = analise3[
    analise3['quantidade_vendida'] >= 15
]

# ============================================
# CLASSE
# ============================================

mediana = analise3['quantidade_vendida'].median()

analise3['sucesso'] = (
    analise3['quantidade_vendida']
    >= mediana
).astype(int)

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

y3 = analise3['sucesso']

# ============================================
# MODELO
# ============================================

modelo3 = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

modelo3.fit(X3, y3)

# ============================================
# IMPORTÂNCIA DAS FEATURES
# ============================================

importancia = pd.DataFrame({
    'Variável': X3.columns,
    'Importância': modelo3.feature_importances_
})

# Ordenar
importancia = importancia.sort_values(
    'Importância',
    ascending=False
)

print("\n🏆 IMPORTÂNCIA DAS VARIÁVEIS:\n")

print(importancia)

# ============================================
# VARIÁVEIS IGNORADAS
# ============================================

ignoradas = importancia[
    importancia['Importância'] == 0
]

print("\n🚫 VARIÁVEIS IGNORADAS:\n")

if len(ignoradas) > 0:

    print(ignoradas['Variável'])

else:

    print("Nenhuma variável foi totalmente ignorada.")

# ============================================
# VISUALIZAÇÃO
# ============================================

plt.figure(figsize=(12,7))

cores = [
    'red' if x == 0 else 'steelblue'
    for x in importancia['Importância']
]

plt.barh(
    importancia['Variável'],
    importancia['Importância'],
    color=cores
)

plt.xlabel('Importância')
plt.ylabel('Variáveis')

plt.title(
    'Pergunta 3: O que não importa?'
)

# Mostrar valores
for i, valor in enumerate(importancia['Importância']):

    plt.text(
        valor,
        i,
        f'{valor:.2f}',
        va='center'
    )

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()

# ============================================
# CONCLUSÃO
# ============================================

print("\n✅ RESPOSTA 3:")

if len(ignoradas) > 0:

    print(
        "A árvore identificou variáveis "
        "que praticamente não ajudam "
        "na tomada de decisão."
    )

else:

    print(
        "Todas as variáveis tiveram "
        "alguma contribuição."
    )
# ============================================
# RANDOM FOREST
# PERGUNTA 1:
# COMO REDUZIR ERROS?
# ============================================

print("\n" + "="*60)
print("🌲 PERGUNTA 1: Como reduzir erros?")
print("="*60)

# ============================================
# IMPORTAÇÕES
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ============================================
# CARREGAR DADOS
# ============================================

df = pd.read_csv(
    "C:\\Users\\Rumanns\\Desktop\\cell phone project analyse\\data\\Amazon_Cell_Phones.csv"
)

print(df.columns)

# ============================================
# PREPARAR DADOS
# ============================================

analise = df.groupby('brand').agg({
    'Price (Dollar)': 'mean',
    'discount_percentage': 'mean',
    'rating_out_of_5': 'mean',
    'RAM (GB)': 'mean',
    'Storage (GB)': 'mean',
    'ID': 'count'
}).reset_index()

# Quantidade vendida
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
# CRIAR CLASSE
# ============================================

mediana = analise['quantidade_vendida'].median()

analise['sucesso'] = (
    analise['quantidade_vendida']
    >= mediana
).astype(int)

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
# DIVIDIR TREINO E TESTE
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# ============================================
# ÁRVORE DE DECISÃO
# ============================================

arvore = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

arvore.fit(X_train, y_train)

pred_arvore = arvore.predict(X_test)

acc_arvore = accuracy_score(
    y_test,
    pred_arvore
)

print(f"\n🌳 Acurácia da Árvore: {acc_arvore:.2f}")

# ============================================
# RANDOM FOREST
# ============================================

floresta = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42
)

floresta.fit(X_train, y_train)

pred_floresta = floresta.predict(X_test)

acc_floresta = accuracy_score(
    y_test,
    pred_floresta
)

print(f"🌲 Acurácia da Floresta: {acc_floresta:.2f}")

# ============================================
# COMPARAÇÃO VISUAL
# ============================================

modelos = [
    'Árvore',
    'Random Forest'
]

acuracias = [
    acc_arvore,
    acc_floresta
]

plt.figure(figsize=(8,6))

plt.bar(
    modelos,
    acuracias
)

plt.ylim(0, 1)

plt.ylabel('Acurácia')

plt.title(
    'Pergunta 1: Como reduzir erros?'
)

# Mostrar valores
for i, valor in enumerate(acuracias):

    plt.text(
        i,
        valor + 0.02,
        f'{valor:.2f}',
        ha='center'
    )

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()

# ============================================
# CONCLUSÃO
# ============================================

if acc_floresta > acc_arvore:

    ganho = (
        acc_floresta - acc_arvore
    ) * 100

    print("\n✅ RESPOSTA 1:")

    print(
        f"O Random Forest reduziu erros "
        f"e melhorou a acurácia em "
        f"{ganho:.1f}%."
    )

else:

    print(
        "\n✅ Os modelos tiveram "
        "desempenho parecido."
    )



# ============================================
# PERGUNTA 2:
# QUEM TOMA A DECISÃO FINAL?
# ============================================

print("\n" + "="*60)
print("🌲 PERGUNTA 2: Quem toma a decisão final?")
print("="*60)

# ============================================
# IMPORTAÇÕES
# ============================================

from collections import Counter

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
# CLASSE
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
# TREINO E TESTE
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X2,
    y2,
    test_size=0.3,
    random_state=42
)

# ============================================
# RANDOM FOREST
# ============================================

floresta2 = RandomForestClassifier(
    n_estimators=10,
    max_depth=4,
    random_state=42
)

floresta2.fit(X_train, y_train)

# ============================================
# PEGAR UMA AMOSTRA
# ============================================

amostra = X_test.iloc[[0]]

print("\n📱 Produto analisado:\n")

print(amostra)

# ============================================
# VOTOS DAS ÁRVORES
# ============================================

votos = []

print("\n🗳️ Votos individuais das árvores:\n")

for i, arvore in enumerate(floresta2.estimators_):

    voto = arvore.predict(amostra)[0]

    votos.append(voto)

    resultado = (
        "SUCESSO"
        if voto == 1
        else "FRACASSO"
    )

    print(f"Árvore {i+1}: {resultado}")

# ============================================
# CONTAGEM DOS VOTOS
# ============================================

contagem = Counter(votos)

print("\n📊 Resultado da votação:\n")

print(contagem)

# ============================================
# DECISÃO FINAL
# ============================================

decisao_final = floresta2.predict(amostra)[0]

resultado_final = (
    "SUCESSO"
    if decisao_final == 1
    else "FRACASSO"
)

print(f"\n🏆 DECISÃO FINAL: {resultado_final}")

# ============================================
# VISUALIZAÇÃO
# ============================================

labels = ['Fracasso', 'Sucesso']

valores = [
    contagem.get(0, 0),
    contagem.get(1, 0)
]

plt.figure(figsize=(7,5))

plt.bar(
    labels,
    valores
)

plt.ylabel('Quantidade de votos')

plt.title(
    'Pergunta 2: Quem toma a decisão final?'
)

# Mostrar valores
for i, valor in enumerate(valores):

    plt.text(
        i,
        valor + 0.2,
        str(valor),
        ha='center'
    )

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()

# ============================================
# CONCLUSÃO
# ============================================

print("\n✅ RESPOSTA 2:")

print(
    "No Random Forest, nenhuma árvore "
    "manda sozinha. A decisão final "
    "vem do consenso entre várias árvores."
)





# ============================================
# PERGUNTA 3:
# QUAIS VARIÁVEIS SOBREVIVEM AO CAOS?
# ============================================

print("\n" + "="*60)
print("🌲 PERGUNTA 3: Quais variáveis sobrevivem ao caos?")
print("="*60)

# ============================================
# IMPORTAÇÕES
# ============================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
# ÁRVORE ÚNICA
# ============================================

arvore3 = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

arvore3.fit(X3, y3)

# ============================================
# RANDOM FOREST
# ============================================

floresta3 = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42
)

floresta3.fit(X3, y3)

# ============================================
# IMPORTÂNCIAS
# ============================================

importancia_arvore = arvore3.feature_importances_

importancia_floresta = floresta3.feature_importances_

# ============================================
# DATAFRAME
# ============================================

comparacao = pd.DataFrame({
    'Variável': X3.columns,
    'Árvore Única': importancia_arvore,
    'Random Forest': importancia_floresta
})

comparacao['Diferença'] = (
    comparacao['Random Forest']
    - comparacao['Árvore Única']
)

comparacao = comparacao.sort_values(
    'Random Forest',
    ascending=False
)

print("\n📊 COMPARAÇÃO DE IMPORTÂNCIA:\n")

print(comparacao)

# ============================================
# VISUALIZAÇÃO
# ============================================

x = np.arange(len(comparacao))

largura = 0.35

plt.figure(figsize=(12,7))

plt.bar(
    x - largura/2,
    comparacao['Árvore Única'],
    largura,
    label='Árvore'
)

plt.bar(
    x + largura/2,
    comparacao['Random Forest'],
    largura,
    label='Random Forest'
)

plt.xticks(
    x,
    comparacao['Variável'],
    rotation=15
)

plt.ylabel('Importância')

plt.title(
    'Pergunta 3: Quais variáveis sobrevivem ao caos?'
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()

# ============================================
# VARIÁVEL MAIS ROBUSTA
# ============================================

topo = comparacao.iloc[0]['Variável']

print("\n🏆 VARIÁVEL MAIS ROBUSTA:")

print(f"{topo}")

# ============================================
# CONCLUSÃO
# ============================================

print("\n✅ RESPOSTA 3:")

print(
    "As variáveis que continuam importantes "
    "mesmo após centenas de árvores "
    "são as mais robustas do modelo."
)
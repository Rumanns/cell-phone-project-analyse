# ============================================
# RESULTADOS
# PERGUNTA 1:
# QUEM ACERTOU MAIS?
# ============================================

print("\n" + "="*60)
print("🏆 PERGUNTA 1: Quem acertou mais?")
print("="*60)

# ============================================
# IMPORTAÇÕES
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ============================================
# CARREGAR DADOS
# ============================================

df = pd.read_csv(
    "C:\\Users\\Rumanns\\Desktop\\cell phone project analyse\\data\\Amazon_Cell_Phones.csv"
)

# ============================================
# PREPARAÇÃO
# ============================================

analise = df.groupby('brand').agg({
    'Price (Dollar)': 'mean',
    'discount_percentage': 'mean',
    'rating_out_of_5': 'mean',
    'RAM (GB)': 'mean',
    'Storage (GB)': 'mean',
    'number_of_ratings': 'sum',
    'ID': 'count'
}).reset_index()

analise.rename(
    columns={'ID': 'quantidade_vendida'},
    inplace=True
)

analise = analise.dropna()

analise = analise[
    analise['quantidade_vendida'] >= 15
]

# ============================================
# CLASSE
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
        'Storage (GB)',
        'number_of_ratings'
    ]
]

y = analise['sucesso']

# ============================================
# NORMALIZAÇÃO
# ============================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ============================================
# TREINO E TESTE
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    random_state=42
)

# ============================================
# MODELOS
# ============================================

modelos = {

    'Regressão': LogisticRegression(),

    'Árvore': DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42
    ),

    'Rede Neural': MLPClassifier(
        hidden_layer_sizes=(10,),
        max_iter=500,
        random_state=42
    )
}

# ============================================
# TREINAR E AVALIAR
# ============================================

resultados = {}

for nome, modelo in modelos.items():

    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)

    acc = accuracy_score(
        y_test,
        pred
    )

    resultados[nome] = acc

    print(f"\n{nome}: {acc:.2f}")

# ============================================
# DATAFRAME
# ============================================

ranking = pd.DataFrame({

    'Modelo': resultados.keys(),

    'Acurácia': resultados.values()

})

ranking = ranking.sort_values(
    'Acurácia',
    ascending=False
)

print("\n📊 RANKING FINAL:\n")

print(ranking)

# ============================================
# VISUALIZAÇÃO
# ============================================

plt.figure(figsize=(10,6))

plt.bar(
    ranking['Modelo'],
    ranking['Acurácia']
)

plt.ylim(0, 1)

plt.ylabel('Acurácia')

plt.title(
    'Pergunta 1: Quem acertou mais?'
)

# Valores
for i, valor in enumerate(ranking['Acurácia']):

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
# CAMPEÃO
# ============================================

campeao = ranking.iloc[0]

print("\n🏆 MODELO CAMPEÃO:")

print(
    f"{campeao['Modelo']} "
    f"com acurácia de "
    f"{campeao['Acurácia']:.2f}"
)

# ============================================
# CONCLUSÃO
# ============================================

print("\n✅ RESPOSTA 1:")

print(
    "O modelo com maior acurácia "
    "foi o que melhor conseguiu "
    "generalizar padrões nos dados."
)



# ============================================
# PERGUNTA 2:
# QUEM EXPLICA MELHOR?
# ============================================

print("\n" + "="*60)
print("🔍 PERGUNTA 2: Quem explica melhor?")
print("="*60)

# ============================================
# INTERPRETABILIDADE
# ============================================

interpretabilidade = {

    'Regressão Linear': 10,

    'Árvore de Decisão': 9,

    'Random Forest': 6,

    'Rede Neural': 2
}

# ============================================
# JUSTIFICATIVAS
# ============================================

explicacoes = {

    'Regressão Linear':
    'Mostra claramente o impacto de cada variável.',

    'Árvore de Decisão':
    'Mostra regras explícitas do tipo SE... ENTÃO.',

    'Random Forest':
    'Mistura centenas de árvores, dificultando a interpretação.',

    'Rede Neural':
    'Aprende padrões internos difíceis de visualizar.'
}

# ============================================
# DATAFRAME
# ============================================

df_interpretacao = pd.DataFrame({

    'Modelo': interpretabilidade.keys(),

    'Interpretabilidade': interpretabilidade.values()

})

df_interpretacao = df_interpretacao.sort_values(
    'Interpretabilidade',
    ascending=False
)

print("\n📊 RANKING DE INTERPRETABILIDADE:\n")

print(df_interpretacao)

# ============================================
# MOSTRAR EXPLICAÇÕES
# ============================================

print("\n🧠 EXPLICAÇÕES:\n")

for modelo, texto in explicacoes.items():

    print(f"{modelo}:")

    print(f"→ {texto}\n")

# ============================================
# VISUALIZAÇÃO
# ============================================

plt.figure(figsize=(10,6))

plt.bar(
    df_interpretacao['Modelo'],
    df_interpretacao['Interpretabilidade']
)

plt.ylim(0, 10)

plt.ylabel('Interpretabilidade')

plt.title(
    'Pergunta 2: Quem explica melhor?'
)

# Valores
for i, valor in enumerate(df_interpretacao['Interpretabilidade']):

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
# MELHOR EXPLICAÇÃO
# ============================================

melhor = df_interpretacao.iloc[0]

print("\n🏆 MODELO MAIS EXPLICÁVEL:")

print(
    f"{melhor['Modelo']}"
)

# ============================================
# CONCLUSÃO
# ============================================

print("\n✅ RESPOSTA 2:")

print(
    "Os modelos mais simples normalmente "
    "são mais fáceis de interpretar. "
    "Já os modelos mais complexos "
    "tendem a funcionar como caixas-pretas."
)



# ============================================
# PERGUNTA 3:
# QUAL MODELO VALE MAIS A PENA?
# ============================================

print("\n" + "="*60)
print("💰 PERGUNTA 3: Qual modelo vale mais a pena?")
print("="*60)

# ============================================
# DADOS DOS MODELOS
# ============================================

modelos = [

    'Regressão Linear',

    'Árvore de Decisão',

    'Random Forest',

    'Rede Neural'
]

# ============================================
# PERFORMANCE
# (usar valores aproximados do vídeo anterior)
# ============================================

performance = [

    0.78,

    0.82,

    0.89,

    0.91
]

# ============================================
# CUSTO COMPUTACIONAL
# (quanto maior, mais pesado)
# ============================================

custo = [

    1,

    2,

    6,

    9
]

# ============================================
# INTERPRETABILIDADE
# ============================================

interpretabilidade = [

    10,

    9,

    6,

    2
]

# ============================================
# SCORE FINAL
# ============================================

score = []

for p, c, i in zip(
    performance,
    custo,
    interpretabilidade
):

    valor = (
        (p * 10)
        + (i * 0.5)
        - (c * 0.4)
    )

    score.append(valor)

# ============================================
# DATAFRAME
# ============================================

ranking_final = pd.DataFrame({

    'Modelo': modelos,

    'Performance': performance,

    'Custo': custo,

    'Interpretabilidade': interpretabilidade,

    'Custo-Benefício': score
})

ranking_final = ranking_final.sort_values(
    'Custo-Benefício',
    ascending=False
)

print("\n📊 RANKING FINAL:\n")

print(ranking_final)

# ============================================
# VISUALIZAÇÃO
# ============================================

plt.figure(figsize=(12,6))

plt.bar(
    ranking_final['Modelo'],
    ranking_final['Custo-Benefício']
)

plt.ylabel('Score')

plt.title(
    'Pergunta 3: Qual modelo vale mais a pena?'
)

# Valores
for i, valor in enumerate(
    ranking_final['Custo-Benefício']
):

    plt.text(
        i,
        valor + 0.2,
        f'{valor:.1f}',
        ha='center'
    )

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()

# ============================================
# CAMPEÃO
# ============================================

campeao = ranking_final.iloc[0]

print("\n🏆 MELHOR CUSTO-BENEFÍCIO:")

print(
    f"{campeao['Modelo']}"
)

# ============================================
# CONCLUSÃO
# ============================================

print("\n✅ RESPOSTA 3:")

print(
    "O melhor modelo não é apenas o que "
    "acerta mais, mas o que entrega "
    "o melhor equilíbrio entre "
    "performance, custo e interpretação."
)

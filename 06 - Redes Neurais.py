# ============================================
# REDES NEURAIS
# PERGUNTA 1:
# COMO A IA APRENDE?
# ============================================

print("\n" + "="*60)
print("🧠 PERGUNTA 1: Como a IA aprende?")
print("="*60)

# ============================================
# IMPORTAÇÕES
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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
# REDE NEURAL
# ============================================

rede = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=300,
    random_state=42
)

# ============================================
# TREINAR REDE
# ============================================

rede.fit(X_train, y_train)

# ============================================
# PREVISÕES
# ============================================

predicoes = rede.predict(X_test)

acuracia = accuracy_score(
    y_test,
    predicoes
)

print(f"\n🎯 Acurácia da rede neural: {acuracia:.2f}")

# ============================================
# LOSS
# ============================================

loss = rede.loss_curve_

# ============================================
# VISUALIZAÇÃO
# ============================================

plt.figure(figsize=(10,6))

plt.plot(loss)

plt.xlabel('Épocas')

plt.ylabel('Erro (Loss)')

plt.title(
    'Pergunta 1: Como a IA aprende?'
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()

# ============================================
# PESOS
# ============================================

print("\n🧠 A rede ajustou pesos internos.")

print(
    f"\nQuantidade de camadas: "
    f"{rede.n_layers_}"
)

print(
    f"Quantidade de neurônios escondidos: "
    f"{rede.hidden_layer_sizes}"
)

# ============================================
# CONCLUSÃO
# ============================================

print("\n✅ RESPOSTA 1:")

print(
    "A rede neural aprende ajustando "
    "pesos internos para reduzir o erro "
    "ao longo do treinamento."
)



# ============================================
# PERGUNTA 2:
# O QUE ACONTECE DENTRO DA REDE?
# ============================================

print("\n" + "="*60)
print("🧠 PERGUNTA 2: O que acontece dentro da rede?")
print("="*60)

# ============================================
# IMPORTAÇÕES
# ============================================

import networkx as nx

# ============================================
# REDE NEURAL
# ============================================

rede2 = MLPClassifier(
    hidden_layer_sizes=(6, 4),
    max_iter=300,
    random_state=42
)

rede2.fit(X_train, y_train)

# ============================================
# ESTRUTURA DA REDE
# ============================================

print("\n📊 Estrutura da rede:\n")

print(f"Camadas: {rede2.n_layers_}")

print(f"Entradas: {X.shape[1]}")

print(f"Camadas escondidas: {rede2.hidden_layer_sizes}")

print("Saída: 1 neurônio")

# ============================================
# VISUALIZAÇÃO DA REDE
# ============================================

G = nx.DiGraph()

# ============================================
# POSIÇÕES DOS NÓS
# ============================================

posicoes = {}

# Entrada
entrada = X.shape[1]

# Camadas escondidas
hidden1 = 6
hidden2 = 4

# Saída
saida = 1

# --------------------------------
# CAMADA DE ENTRADA
# --------------------------------

for i in range(entrada):

    nome = f"E{i}"

    G.add_node(nome)

    posicoes[nome] = (0, -i)

# --------------------------------
# CAMADA ESCONDIDA 1
# --------------------------------

for i in range(hidden1):

    nome = f"H1_{i}"

    G.add_node(nome)

    posicoes[nome] = (1, -i)

# --------------------------------
# CAMADA ESCONDIDA 2
# --------------------------------

for i in range(hidden2):

    nome = f"H2_{i}"

    G.add_node(nome)

    posicoes[nome] = (2, -i)

# --------------------------------
# SAÍDA
# --------------------------------

G.add_node("Saída")

posicoes["Saída"] = (3, -2)

# ============================================
# CONEXÕES
# ============================================

# Entrada -> Hidden 1
for i in range(entrada):

    for j in range(hidden1):

        G.add_edge(
            f"E{i}",
            f"H1_{j}"
        )

# Hidden 1 -> Hidden 2
for i in range(hidden1):

    for j in range(hidden2):

        G.add_edge(
            f"H1_{i}",
            f"H2_{j}"
        )

# Hidden 2 -> Saída
for i in range(hidden2):

    G.add_edge(
        f"H2_{i}",
        "Saída"
    )

# ============================================
# DESENHAR
# ============================================

plt.figure(figsize=(12,8))

nx.draw(
    G,
    posicoes,
    with_labels=True,
    node_size=2000,
    font_size=8,
    arrows=False
)

plt.title(
    'Pergunta 2: O que acontece dentro da rede?'
)

plt.tight_layout()

plt.show()

# ============================================
# PESOS
# ============================================

print("\n🧠 A rede criou conexões internas.")

print(
    "\nCada neurônio recebe informações,"
)

print(
    "combina sinais e passa resultados "
    "para a próxima camada."
)

# ============================================
# CONCLUSÃO
# ============================================

print("\n✅ RESPOSTA 2:")

print(
    "Dentro da rede neural, os dados "
    "passam por várias camadas de neurônios "
    "que transformam informações simples "
    "em padrões mais complexos."
)



# ============================================
# PERGUNTA 3:
# QUANDO A IA COMEÇA A IMAGINAR PADRÕES?
# ============================================

print("\n" + "="*60)
print("🧠 PERGUNTA 3: Quando a IA começa a imaginar padrões?")
print("="*60)

# ============================================
# IMPORTAÇÕES
# ============================================

from sklearn.metrics import accuracy_score

# ============================================
# REDE SIMPLES
# ============================================

rede_simples = MLPClassifier(
    hidden_layer_sizes=(8,),
    max_iter=300,
    random_state=42
)

rede_simples.fit(X_train, y_train)

# ============================================
# REDE COMPLEXA
# ============================================

rede_complexa = MLPClassifier(
    hidden_layer_sizes=(100, 100, 100),
    max_iter=1000,
    random_state=42
)

rede_complexa.fit(X_train, y_train)

# ============================================
# PREVISÕES
# ============================================

# Rede simples
pred_train_simples = rede_simples.predict(X_train)

pred_test_simples = rede_simples.predict(X_test)

# Rede complexa
pred_train_complexa = rede_complexa.predict(X_train)

pred_test_complexa = rede_complexa.predict(X_test)

# ============================================
# ACURÁCIAS
# ============================================

acc_train_simples = accuracy_score(
    y_train,
    pred_train_simples
)

acc_test_simples = accuracy_score(
    y_test,
    pred_test_simples
)

acc_train_complexa = accuracy_score(
    y_train,
    pred_train_complexa
)

acc_test_complexa = accuracy_score(
    y_test,
    pred_test_complexa
)

# ============================================
# RESULTADOS
# ============================================

print("\n📊 REDE SIMPLES")

print(f"Treino: {acc_train_simples:.2f}")

print(f"Teste : {acc_test_simples:.2f}")

print("\n📊 REDE COMPLEXA")

print(f"Treino: {acc_train_complexa:.2f}")

print(f"Teste : {acc_test_complexa:.2f}")

# ============================================
# VISUALIZAÇÃO
# ============================================

modelos = [
    'Simples\nTreino',
    'Simples\nTeste',
    'Complexa\nTreino',
    'Complexa\nTeste'
]

acuracias = [
    acc_train_simples,
    acc_test_simples,
    acc_train_complexa,
    acc_test_complexa
]

plt.figure(figsize=(10,6))

plt.bar(
    modelos,
    acuracias
)

plt.ylim(0, 1)

plt.ylabel('Acurácia')

plt.title(
    'Pergunta 3: Quando a IA começa a imaginar padrões?'
)

# Valores
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
# DETECTAR OVERFITTING
# ============================================

diferenca = (
    acc_train_complexa
    - acc_test_complexa
)

print("\n🧠 DIFERENÇA ENTRE TREINO E TESTE:")

print(f"{diferenca:.2f}")

# ============================================
# CONCLUSÃO
# ============================================

if diferenca > 0.15:

    print("\n⚠️ OVERFITTING DETECTADO!")

    print(
        "A rede aprendeu tão bem os dados "
        "de treino que começou a perder "
        "capacidade de generalização."
    )

else:

    print(
        "\n✅ A rede ainda consegue "
        "generalizar bem."
    )

print("\n✅ RESPOSTA 3:")

print(
    "A IA começa a imaginar padrões "
    "quando aprende detalhes específicos "
    "demais dos dados de treino."
)

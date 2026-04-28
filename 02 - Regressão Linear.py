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
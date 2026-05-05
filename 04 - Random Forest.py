# ============================================
# PERGUNTA 4:
# O QUE MANDA MAIS?
# ============================================

print("\n" + "="*60)
print("🎯 PERGUNTA 4: O que mais impacta as vendas?")
print("="*60)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ============================================
# PREPARAR DADOS
# ============================================

analise4 = df.groupby('brand').agg({
    'Price (Dollar)': 'mean',
    'discount_percentage': 'mean',
    'rating_out_of_5': 'mean',
    'RAM (GB)': 'mean',
    'Storage (GB)': 'mean',
    'number_of_ratings': 'sum',
    'ID': 'count'
}).reset_index()

# Renomear quantidade vendida
analise4.rename(columns={'ID': 'quantidade_vendida'}, inplace=True)

# Remover valores nulos
analise4 = analise4.dropna()

# Filtrar marcas relevantes
analise4 = analise4[analise4['quantidade_vendida'] >= 15]

# ============================================
# VARIÁVEIS
# ============================================

X4 = analise4[
    [
        'Price (Dollar)',
        'discount_percentage',
        'rating_out_of_5',
        'RAM (GB)',
        'Storage (GB)',
        'number_of_ratings'
    ]
]

Y4 = analise4['quantidade_vendida']

# ============================================
# PADRONIZAÇÃO
# ============================================

scaler = StandardScaler()

X4_scaled = scaler.fit_transform(X4)

# ============================================
# MODELO
# ============================================

modelo4 = LinearRegression()

modelo4.fit(X4_scaled, Y4)

# Coeficientes Beta
betas = modelo4.coef_

# ============================================
# ORGANIZAR IMPORTÂNCIA
# ============================================

importancia = pd.DataFrame({
    'Variável': X4.columns,
    'Beta': betas
})

# Magnitude absoluta
importancia['Impacto Absoluto'] = importancia['Beta'].abs()

# Ordenar
importancia = importancia.sort_values(
    'Impacto Absoluto',
    ascending=False
)

print("\n🏆 RANKING DAS VARIÁVEIS MAIS IMPORTANTES:\n")

print(importancia[['Variável', 'Beta']])

# ============================================
# GRÁFICO
# ============================================

plt.figure(figsize=(12,7))

cores = [
    'green' if x > 0 else 'red'
    for x in importancia['Beta']
]

plt.barh(
    importancia['Variável'],
    importancia['Beta'],
    color=cores
)

plt.axvline(x=0, color='black', linestyle='--')

plt.xlabel('Coeficiente Padronizado (Beta)')
plt.ylabel('Variáveis')

plt.title(
    'Pergunta 4: Qual variável manda mais?'
)

# Mostrar valores
for i, valor in enumerate(importancia['Beta']):

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

variavel_topo = importancia.iloc[0]['Variável']

print("\n✅ RESPOSTA 4:")
print(
    f"A variável com maior poder de impacto foi: "
    f"{variavel_topo}"
)
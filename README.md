# cell-phone-project-analyse

Este projeto vai além das estatísticas descritivas básicas (média de preço, avaliação mais comum). Utilizamos uma abordagem de pensamento crítico e diagnóstico de dados para extrair **valor real** do dataset de Smartphones da Amazon.

Em vez de simplesmente listar o que o dado mostra superficialmente, nós o interrogamos com cinco perguntas projetadas para expor armadilhas estatísticas e oportunidades escondidas:

1.  **A Ilusão da Métrica:** 📊 *"Quais dados parecem dar resultados muito bons, ou muito ruins, mas na verdade são o oposto?"*
    - **O que fizemos:** Analisamos a diferença entre o **Valor Aparente vs. Valor Real**. Por exemplo: Um produto com 5 estrelas mas apenas 2 avaliações *parece* perfeito, mas é estatisticamente frágil. Ou um preço baixo que esconde um frete caríssimo. Focamos em desmascarar médias compostas enganosas.

2.  **Fora da Curva (e Por Quê?):** 🚀 *"O que é completamente fora da curva? E por que aconteceu?"*
    - **O que fizemos:** Aplicamos **Detecção de Anomalias e Análise de Resíduos**. Identificamos os smartphones cujo preço não condiz com a especificação técnica ou cujo número de vendas destoa totalmente da média da marca. Investigamos se a causa era um erro de scraping, uma promoção relâmpago ou um modelo de nicho específico.

3.  **Coincidência ou Causalidade Oculta:** 🔗 *"O que acontece ao mesmo tempo? Será que um realmente causa o outro ou tem um terceiro fator escondido?"*
    - **O que fizemos:** Investigamos **Correlações Espúrias e Variáveis de Confusão**. Exemplo clássico esperado: *"Mais RAM = Preço Maior"* (correlação real). Mas e *"Mais Reviews = Maior Nota"*? Será que é qualidade ou efeito manada? Buscamos o **Terceiro Fator Escondido** (ex: a *Marca* ou o *Ano de Lançamento*) que explica as duas variáveis simultaneamente.

4.  **A Alavanca do Negócio:** 💰 *"Qual ponto realmente faz uma diferença palpável? (Agora sim entendi!)"*
    - **O que fizemos:** **Análise de Alavancagem e Feature Importance preliminar**. Identificamos se a diferença entre um celular de R$ 1.000 e R$ 2.000 é só memória ou se envolve um salto tecnológico (ex: de Tela LCD para AMOLED). O foco é encontrar o **ponto de inflexão** onde o valor percebido muda radicalmente.

5.  **Contexto é Rei:** 🌎 *"O que funciona num contexto mas não funciona em outro?"*
    - **O que fizemos:** **Segmentação e Análise de Interação entre Variáveis**. Uma tela de 6.7" é um *diferencial* ou um *defeito*? Descobrimos que a resposta depende da **Estratificação**: Para a categoria *Gamer*, é positivo; para a categoria *Custo-Benefício Compacto*, pode ser irrelevante ou negativo. Separamos os dados por faixa de preço e marca para ver como as "regras" mudam.

---

### 🧠 Metodologia do Projeto
O código foi estruturado para responder essas cinco perguntas utilizando Python (Pandas, Matplotlib, Seaborn) e testes estatísticos específicos para evitar conclusões precipitadas baseadas apenas em visualizações bonitas.


Link dos Dados
https://www.kaggle.com/datasets/michaelmatta0/amazon-cell-phones-cleaned-scraped-data


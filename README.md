# cell-phone-project-analyse

📁 Códigos do Projeto

01 - Preparação do Ambiente
Carrega o dataset de celulares da Amazon, trata colunas numéricas (preço, desconto, avaliação, RAM, armazenamento) e agrupa os dados por marca e modelo para as análises seguintes.

02 - Regressão Linear
Mede numericamente a relação entre preço e quantidade vendida (elasticidade), calcula o desconto ideal (ponto de máximo da curva quadrática) e descobre quanto cada estrela de avaliação "vale" em impacto nas vendas.

03 - Árvore de Decisão
Cria regras do tipo "SE preço < X E avaliação > Y ENTÃO sucesso" para classificar celulares. Mostra combinações específicas de marca, preço e desconto que levam ao fracasso.

04 - Random Forest
Calcula a importância de cada característica do celular (preço, desconto, avaliação, RAM, marca) para prever vendas. Identifica quais variáveis realmente importam e quais podem ser ignoradas.

05 - Redes Neurais
Tenta prever o sucesso de um celular antes do lançamento usando padrões complexos dos dados históricos. Aplicável para recomendação e detecção precoce de tendências.


Link dos Dados
https://www.kaggle.com/datasets/michaelmatta0/amazon-cell-phones-cleaned-scraped-data
# fiap-tech-challenge-001-grupo-16

## Descrição
Este repositório contém os materiais relacionados ao Tech Challenge 001 da FIAP, realizado pelo Grupo 16. O Tech Challenge engloba os conhecimentos obtidos em todas as disciplinas dessa fase. A atividade, obrigatória e valendo 90% da nota de todas as disciplinas da fase, deve ser desenvolvida em grupo e entregue dentro do prazo estipulado.

## Base de Dados
A base de dados para este desafio contém informações sobre custos médicos de indivíduos, incluindo as seguintes colunas:
- `idade`: Idade do beneficiário.
- `gênero`: Gênero do beneficiário.
- `imc`: Índice de Massa Corporal.
- `filhos`: Número de filhos/dependentes cobertos pelo plano de saúde.
- `fumante`: Indica se o beneficiário é fumante.
- `região`: Região de residência do beneficiário.
- `encargos`: Custos médicos individuais cobrados pelo plano de saúde.

Exemplo de dados:
        
idade,gênero,imc,filhos,fumante,região,encargos
56,feminino,29.774373714007336,2,sim,sudoeste,31109.889763423336
46,masculino,25.857394655216346,1,não,nordeste,26650.702646642694
32,masculino,23.014839993647488,0,não,sudoeste,21459.03799039332



## Tarefas
### Exploração de Dados
- Carregar a base de dados e explorar suas características.
- Analisar estatísticas descritivas e visualizar distribuições relevantes.

### Pré-processamento de Dados
- Limpar os dados, tratando valores ausentes se necessário.
- Converter variáveis categóricas para formatos adequados para modelagem.

### Modelagem
- Criar um modelo preditivo de regressão utilizando uma técnica à escolha (ex: Regressão Linear, Árvores de Decisão).
- Dividir o conjunto de dados em conjuntos de treinamento e teste.

### Treinamento e Avaliação do Modelo
- Treinar o modelo com o conjunto de treinamento.
- Validar a eficácia do modelo utilizando métricas estatísticas (p-value, intervalos de confiança).

### Resultados
- Apresentar resultados visuais, como gráficos de previsões vs. valores reais.
- Elaborar um relatório com análise dos resultados, insights obtidos e validação estatística.

## Integrantes do Grupo 16
- Diego Batista Pereira da Silva - alpha.diegobatista@gmail.com
- Samuel Kazuo Watanabe - kazuo_w@hotmail.com
- Jonathan Maximo da Silva - jonathan.desenv@gmail.com
- Eric Pimentel da Silva - ericpimenteldasilva@gmail.com
- Samuel Rodrigues de Barros Mesquita Neto - samuelr.neto98@gmail.com

## Link para o projeto no Colab
[Projeto no Google Colab](https://colab.research.google.com/drive/1zULzY9aOSaV3e-BQA-O_05fROMLyIU5b?usp=sharing)


eda.py
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dados originais
dados_originais = {
    'idade': [56, 46, 32],
    'genero': ['feminino', 'masculino', 'masculino'],
    'imc': [29.774, 25.857, 23.015],
    'filhos': [2, 1, 0],
    'fumante': ['sim', 'não', 'não'],
    'regiao': ['sudoeste', 'nordeste', 'sudoeste'],
    'encargos': [31109.89, 26650.70, 21459.04]
}

# Variável para controlar a quantidade de dados gerados
quantidade_dados = 200

# Gerando dados adicionais com controle de quantidade
np.random.seed(42)  # Para reprodutibilidade

idades_novas = np.random.randint(18, 65, size=quantidade_dados)
generos_novos = np.random.choice(['feminino', 'masculino'], size=quantidade_dados)
imcs_novos = np.random.normal(28, 4, size=quantidade_dados)  # Gerando IMCs com distribuição normal
filhos_novos = np.random.randint(0, 5, size=quantidade_dados)
fumantes_novos = np.random.choice(['sim', 'não'], size=quantidade_dados, p=[0.2, 0.8])  # Probabilidades para fumantes
regioes_novas = np.random.choice(['sudoeste', 'nordeste', 'sudeste', 'norte'], size=quantidade_dados)

# Base de cálculo para encargos ajustada para incluir efeitos de ser fumante e número de filhos
encargos_novos = 2000 + (idades_novas * 200) + (imcs_novos * 150)  # Base de cálculo para encargos
encargos_novos *= (1 + (filhos_novos * 0.1)) * (1.5 if 'sim' in fumantes_novos else 1)

# Unindo os dados originais com os novos dados gerados
dados_ampliados = {
    'idade': np.concatenate((dados_originais['idade'], idades_novas)),
    'genero': np.concatenate((dados_originais['genero'], generos_novos)),
    'imc': np.concatenate((dados_originais['imc'], imcs_novos)),
    'filhos': np.concatenate((dados_originais['filhos'], filhos_novos)),
    'fumante': np.concatenate((dados_originais['fumante'], fumantes_novos)),
    'regiao': np.concatenate((dados_originais['regiao'], regioes_novas)),
    'encargos': np.concatenate((dados_originais['encargos'], encargos_novos))
}

df_ampliado = pd.DataFrame(dados_ampliados)

# Convertendo variáveis categóricas em numéricas com one-hot encoding
df_encoded_ampliado = pd.get_dummies(df_ampliado, drop_first=True)

# Explorando estatísticas descritivas da base de dados ampliada
print("Visão Geral Estatística dos Dados Após Codificação:")
print(df_encoded_ampliado.describe())

# Visualizando relações com gráficos
sns.pairplot(df_encoded_ampliado)
plt.show()

# Calculando e visualizando correlações na base de dados ampliada
print("\nCorrelações Após Codificação:")
correlacoes_ampliadas = df_encoded_ampliado.corr()
sns.heatmap(correlacoes_ampliadas, annot=True, cmap="coolwarm")
plt.show()

```python



modelagem.py

```python
```python
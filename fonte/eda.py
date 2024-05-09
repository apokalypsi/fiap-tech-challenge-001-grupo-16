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

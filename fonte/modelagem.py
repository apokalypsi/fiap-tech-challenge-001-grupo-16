import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Passo 1: Preparação dos Dados (eda.py)

# Identificando características numéricas e já codificadas (categóricas)
features_numericas = ['idade', 'imc', 'filhos']
features_codificadas = [col for col in df_encoded_ampliado.columns if col not in features_numericas + ['encargos']]

# Preparando o transformador de colunas para escalar as características numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_numericas)
    ], remainder='passthrough')

# Passo 2: Definição do Modelo

modelo_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, learning_rate=0.05, random_state=42)

# Criando um pipeline para pré-processamento e modelagem
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', modelo_xgb)])

# Passo 3: Divisão dos Dados em Treino e Teste

X = df_encoded_ampliado.drop('encargos', axis=1)
y = df_encoded_ampliado['encargos']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Passo 4: Treinamento do Modelo

print("Treinando o modelo...")
pipeline.fit(X_train, y_train)

# Passo 5: Avaliação do Modelo

y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nErro Quadrático Médio (MSE): {mse}")
print(f"Coeficiente de Determinação (R^2): {r2} - Quanto mais próximo de 1, melhor.")

# Passo 6: Visualização dos Resultados

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Encargos Reais')
plt.ylabel('Encargos Previstos')
plt.title('Encargos Reais vs. Previstos com XGBoost')
plt.show()

# Comparando as distribuições dos valores reais e previstos
plt.figure(figsize=(10, 6))
sns.histplot(y_test, color="red", alpha=0.5, label='Real')
sns.histplot(y_pred, color="blue", alpha=0.5, label='Previsto')
plt.xlabel('Encargos')
plt.title('Distribuição dos Encargos: Real vs. Previsto')
plt.legend()
plt.show()

print("\nConclusão: O modelo XGBoost demonstrou ser eficaz para prever os custos médicos com base nas características fornecidas, como mostram os resultados visuais e as métricas de avaliação.")

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Carregar dados
train_data = pd.read_csv('/Users/luryan/Documents/persona_project/titanic_dataset/data/train.csv')

# Selecionar apenas colunas numéricas
numeric_columns = train_data.select_dtypes(include=['int64', 'float64']).columns

# Calcular a matriz de correlação
correlation_matrix = train_data[numeric_columns].corr()

# Exportar a matriz de correlação para um arquivo de texto
correlation_matrix_text = correlation_matrix.to_string()
with open('/Users/luryan/Documents/persona_project/titanic_dataset/data/correlation_matrix.txt', 'w') as file:
    file.write(correlation_matrix_text)

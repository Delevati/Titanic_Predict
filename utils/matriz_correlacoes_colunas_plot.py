import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Leitura dos dados
train_data = pd.read_csv('/Users/luryan/Documents/persona_project/titanic_dataset/data/train.csv')
test_data = pd.read_csv('/Users/luryan/Documents/persona_project/titanic_dataset/data/test.csv')

# Definição das variáveis
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Divisão do conjunto de treino para avaliação da assertividade
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

# Criação de um mapa de calor
combined_data = pd.concat([X, y_val])
correlation_matrix = combined_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# Treinamento do modelo
model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=1)
model.fit(X_train, y_train)

# Previsões no conjunto de validação
predictions_val = model.predict(X_val)

# Avaliação do modelo
accuracy = accuracy_score(y_val, predictions_val)
print("Overall Accuracy:", accuracy)

# Outras métricas
print("\nClassification Report:")
print(classification_report(y_val, predictions_val))

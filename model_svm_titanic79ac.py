import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # Import Imputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Leitura dos dados
train_data = pd.read_csv('/Users/luryan/Documents/persona_project/titanic_dataset/train.csv')
test_data = pd.read_csv('/Users/luryan/Documents/persona_project/titanic_dataset/data/test.csv')

# Pré-processamento dos dados
features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"]
X = pd.get_dummies(train_data[features])
y = train_data["Survived"]

# Tratar valores ausentes
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Divisão do conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=5)

# Padronização dos dados (escalonamento)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinamento do modelo SVM
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train_scaled, y_train)

# Previsões no conjunto de teste
X_test_data = pd.get_dummies(test_data[features])
X_test_data = pd.DataFrame(imputer.transform(X_test_data), columns=X_test_data.columns)
X_test_data_scaled = scaler.transform(X_test_data)
predictions = svm_model.predict(X_test_data_scaled)

# Salvar as previsões em um arquivo CSV
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('./outputs/submission_svm.csv', index=False)

# Avaliação do modelo no conjunto de teste
accuracy = accuracy_score(y_test, svm_model.predict(X_test_scaled))
print("Accuracy:", accuracy)

# Relatório de classificação
print("Classification Report:\n", classification_report(y_test, svm_model.predict(X_test_scaled)))

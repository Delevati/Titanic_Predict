import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Carregar os dados
train_data = pd.read_csv('/Users/luryan/Documents/persona_project/titanic_dataset/data/train.csv')
test_data = pd.read_csv('/Users/luryan/Documents/persona_project/titanic_dataset/data/test.csv')

# Separar os conjuntos de treino e teste
y_train = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"]
X_train = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

# Tratamento de nans
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_train.mean(), inplace=True)

# Concatenar para matriz de correlação
combined_data = pd.concat([X_train, y_train], axis=1)
correlation_matrix = combined_data.corr()

# plt.figure(figsize=(5, 5))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
# penguins = sns.load_dataset("penguins")
# sns.pairplot(train_data, hue="Survived")
# plt.title('PairPlot')
# plt.savefig('/Users/luryan/Documents/persona_project/titanic_dataset/outputs/pairplot.png')
# plt.show()

# Modelo RandomForest
model = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=1)
model.fit(X_train, y_train)

# Fazer predições
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': model.predict(X_test)})
output.to_csv('./outputs/submission_random_forest.csv', index=False)

# Avaliação do modelo
predictions_train = model.predict(X_train)
class_report_train = classification_report(y_train, predictions_train)
accuracy_train = accuracy_score(y_train, predictions_train)

print("Overall Accuracy on Train Data:", accuracy_train)
print("\nClassification Report on Train Data:")
print(class_report_train)

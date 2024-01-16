import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train_data = pd.read_csv('/Users/luryan/Documents/persona_project/titanic_dataset/Data/train_rforest_MF.csv')
test_data = pd.read_csv('/Users/luryan/Documents/persona_project/titanic_dataset/data/test.csv')

features = ["Pclass", "SibSp", "Sex"]

X = train_data[features]
y = train_data["Survived"]

X_train, X_val, y_train, y_val = train_test_split(X, y)
plt.figure(figsize=(8, 6))
sns.heatmap(X_train.corr(), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix')
# plt.show()

knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Não Sobreviveu", "Sobreviveu"], yticklabels=["Não Sobreviveu", "Sobreviveu"])
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

class_report = classification_report(y_val, y_pred)
print("Classification Report:")
print(class_report)

X_test = test_data[features]

if 'Sex' in X_test.columns and X_test['Sex'].dtype == 'object':
    X_test.loc[:, 'Sex'] = X_test['Sex'].map({'male': 0, 'female': 1})

predictions = knn_model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('./outputs/submission_knn.csv', index=False)

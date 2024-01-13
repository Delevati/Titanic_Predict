import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('/Users/luryan/Documents/persona_project/titanic_dataset/data/train_rforest_MF.csv')
test_data = pd.read_csv('/Users/luryan/Documents/persona_project/titanic_dataset/data/test.csv')

y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

#tratamento de nans
X_train.fillna(X_train.mean(), inplace=True)
X_val.fillna(X_train.mean(), inplace=True) 

X_test.fillna(X_train.mean(), inplace=True)

combined_data = pd.concat([X, y_val])
correlation_matrix = combined_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=1)
model.fit(X_train, y_train)

predictions_val = model.predict(X_val)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': model.predict(X_test)})
output.to_csv('./outputs/submission_random_forest.csv', index=False)

class_report = classification_report(y_val, predictions_val)
accuracy = accuracy_score(y_val, predictions_val)

print("Overall Accuracy:", accuracy)
print("\nClassification Report:")
print(class_report)

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

train_data = pd.read_csv('/Users/luryan/Documents/persona_project/titanic_dataset/data/train.csv')
test_data = pd.read_csv('/Users/luryan/Documents/persona_project/titanic_dataset/data/test.csv')

y = train_data["Survived"]
X = train_data
y_train = X["Survived"]
X.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

colunas_categoricas = ['Sex', "Embarked"]

one_hot_enc = make_column_transformer(
    (OneHotEncoder(handle_unknown = 'ignore'),
    colunas_categoricas),
    remainder='passthrough')

X = one_hot_enc.fit_transform(X)
X = pd.DataFrame(X, columns=one_hot_enc.get_feature_names_out())
X = X.fillna(X.median())
# X_test = pd.get_dummies(test_data[features])

# X, X_val, y_train, y_val = train_test_split(X, y)

#tratamento de nans
# X_val.fillna(X.mean(), inplace=True) 

# X_test.fillna(X.mean(), inplace=True)

# combined_data = pd.concat([X, y_val])
# correlation_matrix = combined_data.corr()

# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
# plt.title('Correlation Matrix')
# plt.show()
plt.figure(figsize=(5, 5))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
# penguins = sns.load_dataset("penguins")
sns.pairplot(X, hue="Survived")
plt.title('PairPlot')
plt.savefig('/Users/luryan/Documents/persona_project/titanic_dataset/outputs/pairplot.png')

# model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=1)
# model.fit(X, y_train)

# X_val = test_data
# X_val.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)
# # X_val.dropna(inplace=True)
# # X_val.fillna(X_val.mean(), inplace=True) 
# X_val = X_val.fillna(X_val.median())
# X_val = one_hot_enc.fit_transform(X_val)
# X_val = pd.DataFrame(X_val, columns=one_hot_enc.get_feature_names_out())

# predictions_val = model.predict(X_val)

# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': model.predict(X_val)})
# output.to_csv('./outputs/submission.csv', index=False)

# # class_report = classification_report(y_val, predictions_val)
# # accuracy = accuracy_score(y_val, predictions_val)

# # print("Overall Accuracy:", accuracy)
# # print("\nClassification Report:")
# # print(class_report)

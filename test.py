import pandas as pd

# Importing the data
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score

data_train = pd.read_csv('Data/train.csv')
data_test = pd.read_csv('Data/test.csv')

# print(data_train.head())

# # check null
# print(data_train.isnull().sum())
# print(data_test.isnull().sum())

# # Data visualization
# sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train)
# plt.show()

# fill na by median of ages
impute_value = data_train['Age'].median()
data_train['Age'] = data_train['Age'].fillna(impute_value)
data_test['Age'] = data_test['Age'].fillna(impute_value)

# add isFemale column
data_train['IsFemale'] = (data_train['Sex'] == 'female').astype(int)
data_test['IsFemale'] = (data_test['Sex'] == 'female').astype(int)

predictors = ['Pclass', 'IsFemale', 'Age']
X_train = data_train[predictors].values
X_test = data_test[predictors].values
y_train = data_train['Survived'].values
# y_test = data_test['Survived'].values

# print(X_train[:5])

# train model using Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# train model using Logistic Regression CV
model_cv = LogisticRegressionCV()
model_cv.fit(X_train, y_train)

# predict
y_predict = model.predict(X_test)
print(y_predict[:10])

# calculate scores
scores = cross_val_score(model, X_train, y_train, cv=4)
print(scores)

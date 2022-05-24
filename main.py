import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# LOAD DATA

data = pd.read_csv(r"/root/Downloads/heart_failure_clinical_records_dataset.csv")

# data exploration

print(len(data))
print(data.shape)
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

# EXPLORATION DATA ANALYSIS

"""
create 2 classes that helps in visualising the target data 
to answer  questions on the data by visualization
"""
Live = len(data["DEATH_EVENT"][data.DEATH_EVENT == 0])
Death = len(data["DEATH_EVENT"][data.DEATH_EVENT == 1])

arr = np.array([Live, Death])
labels = ['Living', 'Dead']

print(f"Total living cases : {Live}")
print(f"Total Death cases : {Death}")

# plot a pie chart
plt.pie(arr, labels=labels, explode=[0.2, 0.0], shadow=True)
plt.show()

# imbalance data not distributed equally in classes
# this project is an imbalances data 203 : 96


# visualizing the distribution on the features
sn.displot(data["age"])
plt.show()

sn.displot(data["age"])
plt.show()

sn.displot(data["creatinine_phosphokinase"])
plt.show()

sn.displot(data["ejection_fraction"])
plt.show()

sn.displot(data["platelets"])
plt.show()

sn.displot(data["serum_creatinine"])
plt.show()

# age above 45

age_live = len(data["DEATH_EVENT"][data.age >= 45][data.DEATH_EVENT == 0])
age_death = len(data["DEATH_EVENT"][data.age >= 45][data.DEATH_EVENT == 1])

arr1 = np.array([age_live, age_death])
labels = ["Died", 'Alive']

print(f"TOtal died at 45 :{age_death}")
print(f"TOtal alive at 45 :{age_live}")

# plot a pie chart
plt.pie(arr, labels=labels, explode=[0.2, 0.0], shadow=True)
plt.show()

# if patient have  diabetes and have died

present = len(data["DEATH_EVENT"][data.diabetes == 1][data.DEATH_EVENT == 1])
absent = len(data["DEATH_EVENT"][data.diabetes == 1][data.DEATH_EVENT == 0])

arr2 = np.array([absent, present])
labels = ["Died", 'Alive']

print(f"TOtal died with diabities :{present}")
print(f"TOtal alive  :{absent}")

# plot a pie chart
plt.pie(arr, labels=labels, explode=[0.2, 0.0], shadow=True)
plt.show()

# CORRELATION OF THE DATA

"""
Help to remove features that are less valuable
"""

co_r = data.corr()
plt.subplots(figsize=(15, 15))
sn.heatmap(co_r, annot=True)
plt.show()

# SPLIT THE DATA

x = data.drop("DEATH_EVENT", axis=1)
y = data["DEATH_EVENT"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# BUILD THE MODEL

"""
TODO this section is not printing out
"""


def evaluate(y_test, y_pred):
    print(f"acc score : {accuracy_score(y_test, y_pred)}")
    print(f"Precision score : {precision_score(y_test, y_pred)}")
    print(f"recall score : {recall_score(y_test, y_pred)}")
    print(f"Confusion matrix : \n {confusion_matrix(y_test, y_pred)}")


lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)

lr_pred = lr.predict(x_test)

y_pred = lr_pred

evaluate(y_test, y_pred)

lr_pip = make_pipeline(StandardScaler(), LogisticRegression())
lr_pip.fit(x_train, y_train)

y_pred1 = lr_pip.predict(x_test)
evaluate(y_test, y_pred1)

# BUILD A SUPPORT VECTOR CLASSIFIER

# define parameter range
param_grid = {"C": [0.1, 1, 100, 1000],
              "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
              "kernel": ["rbf"]}
grid = GridSearchCV(SVC(), param_grid=param_grid, refit=True, verbose=3)
grid.fit(x_train, y_train)

# best estimator
est = grid.best_estimator_
print(est)

# using the best estimate
svc = SVC(C=100, gamma=0.0001)
svc.fit(x_train, y_train)
y_pred2 = svc.predict(x_test)
evaluate(y_test, y_pred2)

"""
The SVC will perform less than the logistic regression
"""


# BUILD A DECISION TREE CLASSIFIER

def random_search(params, runs=20, clf=DecisionTreeClassifier(random_state=2)):
    random = RandomizedSearchCV(clf, params, n_iter=runs, cv=5, n_jobs=1, random_state=2)
    random.fit(x_train, y_train)
    best_model = random.best_estimator_
    score = random.best_score_

    print(f"Training score {format(score)}")
    pred3 = best_model.predict(x_test)
    acc = accuracy_score(y_test, pred3)
    print("Test score:{:.3f}".format(acc))

    print(best_model)


random_search(params={'criterion': ['entropy', 'gini'],
                      'splitter': ['random', 'best'],
                      'min_weight_fraction_leaf': [0.0, 0.0025, 0.005, 0.0075, 0.01],
                      'min_samples_split': [2, 3, 4, 5, 6, 8, 10],
                      'min_samples_leaf': [1, 0.01, 0.02, 0.03, 0.04],
                      'min_impurity_decrease': [0.0, 0.0005, 0.005, 0.05, 0.10, 0.15, 0.2],
                      'max_leaf_nodes': [10, 15, 20, 25, 30, 35, 40, 45, 50, None],
                      'max_features': ['auto', 0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
                      'max_depth': [None, 2, 4, 6, 8],
                      'min_weight_fraction_leaf': [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.05]
                      })

"""
From this we can obtain the best model for fine tunning the decision tree
"""

# create the model ith the result
dt = DecisionTreeClassifier(max_depth=4, max_features=0.75, max_leaf_nodes=40,
                            min_impurity_decrease=0.05, min_samples_leaf=0.02,
                            min_samples_split=8, min_weight_fraction_leaf=0.0025,
                            random_state=2)

dt.fit(x_train, y_train)
predict = dt.predict(x_test)
evaluate(y_test, predict)

# RANDOM FOREST CLASSIFIER
random_search(params={'min_samples_leaf': [1, 2, 4, 6, 8, 10, 20, 30],
                      'min_impurity_decrease': [0.0, 0.01, 0.05, 0.10, 0.15, 0.2],
                      'max_features': ['auto', 0.8, 0.7, 0.6, 0.5, 0.4],
                      'max_depth': [None, 2, 4, 6, 8, 10, 20],
                      },
              clf=RandomForestClassifier(random_state=2))

rt = RandomForestClassifier(max_depth=2, max_features=0.8, min_impurity_decrease=0.1,
                            random_state=2)
rt.fit(x_train, y_train)
predict1 = rt.predict(x_test)
evaluate(y_test, predict1)


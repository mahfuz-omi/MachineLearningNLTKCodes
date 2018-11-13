from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

df = pd.read_csv("thesis_data2.csv")
pd.set_option('expand_frame_repr', False)


# separate array into input and output components
X = df.values[:,1:5]
y = df.values[:,5]
minMaxScaler = MinMaxScaler()
reScaledX = minMaxScaler.fit_transform(X)

kfold = model_selection.KFold(n_splits=10)
avg_accuracy_score = 0
for train_index, test_index in kfold.split(reScaledX,y):
    X_train, X_test = reScaledX[train_index], reScaledX[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = SVC()
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    confusion_matrix_val = confusion_matrix(y_test, y_predicted)
    print(confusion_matrix_val)
    avg_accuracy_score += accuracy_score(y_test, y_predicted)
    print(accuracy_score(y_test, y_predicted))

print("avg accuracy score :",avg_accuracy_score)


# results = model_selection.cross_val_score(model, reScaledX, y, cv=kfold)
# avg_score = np.mean(results)

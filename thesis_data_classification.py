# data normalization (min/max scaler)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score

df = pd.read_csv("thesis_data2.csv")
pd.set_option('expand_frame_repr', False)


# separate array into input and output components
X = df.values[:,1:5]
y = df.values[:,5]
minMaxScaler = MinMaxScaler()
reScaledX = minMaxScaler.fit_transform(X)

# summarize transformed data
np.set_printoptions(precision=3)
print(reScaledX)

df_new = pd.DataFrame(reScaledX)
print(df_new.head(15))

# k-fold+logistic_regression+accuracy_score

from sklearn import model_selection
#
kfold = model_selection.KFold(n_splits=10)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=20)
results = model_selection.cross_val_score(model, reScaledX, y, cv=kfold)
avg_score = np.mean(results)
print("Avg. Accuracy Score for KNeighborsClassifier:",round(avg_score, 3))
#
#
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
results = model_selection.cross_val_score(model, reScaledX, y, cv=kfold)
avg_score = np.mean(results)
print("Avg. Accuracy Score for DecisionTreeClassifier:",round(avg_score, 3))
#
#
from sklearn.svm import SVC
model = SVC(kernel='linear')
results = model_selection.cross_val_score(model, reScaledX, y, cv=kfold)
avg_score = np.mean(results)
print("Avg. Accuracy Score for SVC:",round(avg_score, 3))
#
#
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
results = model_selection.cross_val_score(model, reScaledX, y, cv=kfold)
avg_score = np.mean(results)
print("Avg. Accuracy Score for RandomForestClassifier:",round(avg_score, 3))
#
#
from sklearn.neural_network import MLPClassifier
# total 3 hidden layers, each having 10 number of neurons
model = MLPClassifier(hidden_layer_sizes=(10,10,10))
results = model_selection.cross_val_score(model, reScaledX, y, cv=kfold)
avg_score = np.mean(results)
print("Avg. Accuracy Score for MLPClassifier:",round(avg_score, 3))


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
results = model_selection.cross_val_score(model, reScaledX, y, cv=kfold)
avg_score = np.mean(results)
print("Avg. Accuracy Score for GaussianNB:",round(avg_score, 3))
#
#
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

def convert_to_binary(x):
    if(x == "Depressed"):
        return 1
    elif(x == "Not-Depressed"):
        return 0

df['Mental State(Depressed/Not-depressed)'] = df.apply(lambda row: convert_to_binary(row['Mental State(Depressed/Not-depressed)']), axis=1)
df['Mental State(Depressed/Not-depressed)'].astype(int)
#
results = model_selection.cross_val_score(model, reScaledX, df["Mental State(Depressed/Not-depressed)"].values, cv=kfold)
avg_score = np.mean(results)
print("Avg. Accuracy Score for LogisticRegression:",round(avg_score, 3))

def convert_to_categorical(x):
    if(x == 1):
        return "Depressed"
    elif(x == 0):
        return "Not-Depressed"

#df['Mental State(Depressed/Not-depressed'] = df.apply(lambda row: convert_to_categorical(row['Mental State(Depressed/Not-depressed']), axis=1)




# confusion matrix
from sklearn.metrics import confusion_matrix
cm_KNeighborsClassifier = 0
cm_DecisionTreeClassifier = 0
cm_SVC = 0
cm_RandomForestClassifier = 0
cm_MLPClassifier = 0
cm_GaussianNB = 0
cm_LogisticRegression = 0

kfold = model_selection.KFold(n_splits=10)
for train_index, test_index in kfold.split(df):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = KNeighborsClassifier(n_neighbors=20)
    model.fit(X_train,y_train)
    y_predicted = model.predict(X_test)
    cm_KNeighborsClassifier += confusion_matrix(y_test,y_predicted)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    cm_DecisionTreeClassifier += confusion_matrix(y_test, y_predicted)

    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    cm_SVC += confusion_matrix(y_test, y_predicted)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    cm_RandomForestClassifier += confusion_matrix(y_test, y_predicted)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    cm_MLPClassifier += confusion_matrix(y_test, y_predicted)

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    cm_GaussianNB += confusion_matrix(y_test, y_predicted)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    cm_LogisticRegression += confusion_matrix(y_test, y_predicted)


print("KNeighborsClassifier Confusion_Matrix: ",cm_KNeighborsClassifier)
print("DecisionTreeClassifier Confusion_Matrix: ",cm_DecisionTreeClassifier)
print("SVC Confusion_Matrix: ",cm_SVC)
print("RandomForestClassifier Confusion_Matrix: ",cm_RandomForestClassifier)
print("MLPClassifier Confusion_Matrix: ",cm_MLPClassifier)
print("GaussianNB Confusion_Matrix: ",cm_GaussianNB)
print("LogisticRegression Confusion_Matrix: ",cm_LogisticRegression)


# model = RandomForestClassifier()
# results = model_selection.cross_val_score(model, reScaledX, y, cv=kfold)
# avg_score = np.mean(results)
# print("Avg. Accuracy Score for MLPClassifier:",round(avg_score, 3))
# print(results)








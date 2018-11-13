import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np

df = pd.read_csv('data.csv')
print(df)

# regular system, train-test-split
X = df.drop('age',axis=1)
#print(type(X))
y = df['age']
#print(type(y))

print(X,y)
print(y.values)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print("after train-test split")
print(X_train,X_test,y_test,y_test,sep='\n')

# k-fold cross validation
print('k fold cross')
from sklearn.model_selection import KFold
# prepare cross validation
kfold = KFold(5, True, 1)

# enumerate splits
# train and test hold the corresponding indices as lists
for train, test in kfold.split(df):
    print('train ',train,'test ',test)
    #print('train: %s, test: %s' % (df[train], df[test]))
    train_data = np.array(df)[train]
    test_data = np.array(df)[test]
    print(train_data)
    print(test_data)
    dfnew = pd.DataFrame(train_data,columns=['id','name','age'])
    print(dfnew)
    X = dfnew.drop('age',axis=1).values
    y = dfnew['age'].values
    print(X,y)

    # this X and y is used for the corresponding model



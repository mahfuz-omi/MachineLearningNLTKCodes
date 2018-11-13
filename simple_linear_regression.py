import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


df = pd.read_csv('linear_data.csv')
print(df)
X = df[['age']].values
y = df['income'].values

linearModel = LinearRegression()
linearModel.fit(X,y)

print(linearModel.score(X,y))

dict = {'age':[50,55,60,40]}
X_test = pd.DataFrame(dict)
print('X_test',X_test)

y_prediction = linearModel.predict(X_test)
print(y_prediction)


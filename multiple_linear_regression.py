import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('multiple_linear_data.csv')
print(df)
X = df[['age','merit']].values
y = df['income'].values

linearModel = LinearRegression()
linearModel.fit(X,y)

dict = {'age':[50,55,60,40],'merit':[10,12,20,8]}
X_test = pd.DataFrame(dict)
print('X_test',X_test)

y_prediction = linearModel.predict(X_test.values)
print(y_prediction)
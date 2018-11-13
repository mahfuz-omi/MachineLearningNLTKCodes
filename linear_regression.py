import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# the code has errors, need to resolve
# least square linear regression
from sklearn.linear_model import LinearRegression

df = pd.read_csv('linear_regression.csv')
print(df)


#df.X and df['X'] are equal
# X = df.X
# print(X.values)
# Y = df.Y
# print(Y.values)

# plt.scatter(df.X,df.Y)
# plt.show()
# # print('omi')
# # print(df['X'])
# # print(df['Y'])
#
linearRegression = LinearRegression()
linearRegression.fit(df,df['Y'])
#
resultY = linearRegression.predict(pd.Series([15,16,17,18,19,20]))
plt.scatter([15,16,17,18,19,20],resultY.values)
plt.show()


print(linearRegression.intercept_)
print(linearRegression.coef_)


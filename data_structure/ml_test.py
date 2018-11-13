import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("datas/student_scores.csv")
print(df)

#     Hours  Scores
# 0     2.5      21
# 1     5.1      47
# 2     3.2      27
# 3     8.5      75
# 4     3.5      30
# 5     1.5      20

print(df['Hours'])

# 0     2.5
# 1     5.1
# 2     3.2
# 3     8.5
# 4     3.5
# 5     1.5

print(df['Scores'])

# 0     21
# 1     47
# 2     27
# 3     75
# 4     30
# 5     20

X = df[['Hours']].values  #dataframe values
y = df['Scores'].values # series values

print(X)
print(y)

plt.plot(X,y,"ro")
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test)

print(y_pred)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('petrol_consumption.csv')
print(df.head())

print(df.shape)

plt.plot(df['Paved_Highways'].values,df['Petrol_Consumption'].values)
plt.show()

# show correlations
correlations = df.corr()
print(correlations)
plt.matshow(correlations)
plt.show()


X = df.drop('Petrol_Consumption', axis=1)
y = df['Petrol_Consumption']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(df)


#Evaluating the Algorithm
#To evaluate performance of the regression algorithm,
# the commonly used metrics are mean absolute error,
# mean squared error, and root mean squared error.
# The Scikit-Learn library contains functions that can help calculate these values for us.
# To do so, use this code from the metrics package:

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
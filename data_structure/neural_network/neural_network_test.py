# and data test

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("and_data.csv")
print(df)

plt.plot(df['x1'],df['x2'],"p")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

from sklearn.neural_network import MLPClassifier
# total 3 hidden layers, each having 10 number of neurons
model = MLPClassifier(hidden_layer_sizes=(2))
model.fit(df[['x1','x2']],df['y'])

print(model.intercepts_)
print(model.coefs_)


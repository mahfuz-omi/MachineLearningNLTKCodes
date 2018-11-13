import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('iris.csv')
print(df)

X = df[['sepal_length','sepal_width','petal_length','petal_width']].values
y = df['species'].values

print(X)
print(y)

model = KNeighborsClassifier(n_neighbors=20)
model.fit(X,y)
dict = {'sepal_length':[6.4,0.2,4.9,5.6],'sepal_width':[3.4,0.8,0.7,4.2],'petal_length':[3.0,1.7,6.4,6.7],'petal_width':[1.8,2.2,0.2,0.4]}
X_predict = pd.DataFrame(dict)
print(X_predict.values)

species_pred = model.predict(X_predict)
print(species_pred)
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd

# other options are also available
X1, Y1 = make_classification(n_samples=1000, n_features=4, n_classes=2,n_redundant=0,
    n_repeated=0,n_informative=4,n_clusters_per_class=1,class_sep=3.0,shift=[9.0,9.0,9.0,9.0],scale=[100,20,5,25])

# result of make_classification
print("result of make_classification")
df1 = pd.DataFrame(X1)
df2 = pd.DataFrame(Y1)
df1['result'] = df2
pd.set_option('expand_frame_repr', False)
print(df1)

df1=df1.astype(int)



print("Feature names Modification")
df1.columns = ['Avg. HRV', 'Avg. Outgoing Call Duration','Avg. Outgoing Message length','Avg. Interaction in Social Media','Result']

def valuation_formula(x):
    if(x == 1):
        return "Depressed"
    elif(x == 0):
        return "Not-Depressed"

df1['Result'] = df1.apply(lambda row: valuation_formula(row['Result']), axis=1)

pd.set_option('expand_frame_repr', False)
print(df1)



df1.to_csv('thesis_data.csv')




plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
            s=25, edgecolor='k')

plt.show()
#
# #Build a forest and compute the feature importances
# forest = ExtraTreesClassifier(n_estimators=250,
# #                               random_state=0)
# #
# #
# forest.fit(X1, Y1)
# #
# #
# #array with importances of each feature
# importances = forest.feature_importances_
#
# print(importances)



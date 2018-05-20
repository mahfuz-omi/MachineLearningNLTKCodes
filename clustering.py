import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
df = pd.read_csv('customer2.csv')
# print(df)
print(df['Channel'].values)
df.drop(columns=['Channel','Region'],axis=1,inplace=True)
print(df)


# print(df.loc[[100, 200, 300],:])
# print(df.columns)
# print(df.Frozen)

# find the value of k, the number of cluster, for using in k-means algm
# elbow method
cluster_range = range( 1, 20 )
cluster_errors = []

for num_clusters in cluster_range:
  kmeans = KMeans( num_clusters )
  kmeans.fit(df)
  cluster_errors.append( kmeans.inertia_)


clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
print(clusters_df)

plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
plt.show()

# so, optimal number of cluster is 4
# run the cluster algm for cluster no = 4
# feature scaling is needed, but not done here, left for another day

kmeans = KMeans(4)
kmeans.fit(df)
print(kmeans.cluster_centers_)
print(kmeans.labels_)
# plt.scatter(df['Milk'], c=kmeans.labels_, cmap='rainbow')
# plt.scatter(df['Fresh'], c=kmeans.labels_, cmap='rainbow')
# plt.scatter(df['Grocery'], c=kmeans.labels_, cmap='rainbow')
# plt.scatter(df['Frozen'], c=kmeans.labels_, cmap='rainbow')
# plt.scatter(df['Detergents_Paper'], c=kmeans.labels_, cmap='rainbow')
# plt.scatter(df['Delicassen'], c=kmeans.labels_, cmap='rainbow')
# plt.show()
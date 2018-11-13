import pandas as pd
import matplotlib.pyplot as plt

# url: http://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
df = pd.read_csv('iris.csv')
print(df.head())

X = df.drop('species', 1)
y = df['species']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#PCA performs best with a normalized feature set.
# We will perform standard scalar normalization to normalize our feature set.
# To do this, execute the following code:

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
# It is only a matter of three lines of code to perform PCA using Python's Scikit-Learn library.
# The PCA class is used for this purpose.
# PCA depends only upon the feature set and not the label data.
# Therefore, PCA can be considered as an unsupervised machine learning technique.
#
# Performing PCA using Scikit-Learn is a two-step process:
#
# Initialize the PCA class by passing the number of components to the constructor.
# Call the fit and then transform methods by passing the feature set to these methods.
# The transform method returns the specified number of principal components.
# Take a look at the following code:

from sklearn.decomposition import PCA

pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

#The PCA class contains explained_variance_ratio_ which returns the variance
# caused by each of the principal components.
# Execute the following line of code to find the "explained variance ratio".

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

#The explained_variance variable is now a float type array which contains variance ratios for each principal component. The values for the explained_variance variable looks like this:

#It can be seen that first principal component is responsible for 72.22% variance.
# Similarly, the second principal component causes 23.9% variance in the dataset.
# Collectively we can say that (72.22 + 23.9) 96.21% percent of the classification information contained
# in the feature set is captured by the first two principal components.

#Let's first try to use 1 principal component to train our algorithm. To do so, execute the following code:

from sklearn.decomposition import PCA

pca = PCA(n_components=1)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
#The rest of the process is straight forward.

# Training and Making
# Predictions
# In
# this
# case
# we
# 'll use random forest classification for making the predictions.

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Performance
# Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy' , accuracy_score(y_test, y_pred))
# The
# output
# of
# the
# script
# above
# looks
# like
# this:

# [[11  0  0]
#  [0 12  1]
# [0
# 1
# 5]]
# 0.933333333333
# It
# can
# be
# seen
# from the output
#
# that
# with only one feature, the random forest algorithm is able to correctly predict 28 out of 30 instances, resulting in 93.33 % accuracy.
#
# Results
# with 2 and 3 Principal Components
# Now
# let
# 's try to evaluate classification performance of the random forest algorithm with 2 principal components. Update this piece of code:

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
# Here
# the
# number
# of
# components
#for PCA has been set to 2. The classification results with 2 components are as follows:

# [[11  0  0]
#  [0 10  3]
# [0
# 2
# 4]]
# 0.833333333333
# With
# two
# principal
# components
# the
# classification
# accuracy
# decreases
# to
# 83.33 % compared
# to
# 93.33 %
# for 1 component.
#
# With
# three
# principal
# components, the
# result
# looks
# like
# this:
#
# [[11  0  0]
#  [0 12  1]
# [0
# 1
# 5]]
# 0.933333333333
# With
# three
# principal
# components
# the
# classification
# accuracy
# again
# increases
# to
# 93.33 %
#
# Results
# with Full Feature Set
# Let
# 's try to find the results with full feature set. To do so, simply remove the PCA part from the script that we wrote above. The results with full feature set, without applying PCA looks like this:
#
# [[11  0  0]
#  [0 13  0]
# [0
# 2
# 4]]
# 0.933333333333
# The
# accuracy
# received
# with full feature set is for random forest algorithm is also 93.33 %.
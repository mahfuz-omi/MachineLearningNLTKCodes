import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# url: http://stackabuse.com/decision-trees-in-python-with-scikit-learn/

df = pd.read_csv('bill_authentication.csv')
print(df)

print(df.shape)

#To divide data into attributes=X and labels=y, execute the following code:
X = df.drop('Class', axis=1)
y = df['Class']
print(X)
print(y)
#Here the X variable contains all the columns from the dataset,
# except the "Class" column, which is the label.
# The y variable contains the values from the "Class" column.
# The X variable is our attribute set and y variable contains corresponding labels.
# The final preprocessing step is to divide our data into training and test sets.
# The model_selection library of Scikit-Learn contains train_test_split method,
#  which we'll use to randomly split the data into training and testing sets.
# Execute the following code to do so:
#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
#In the code above, the test_size parameter specifies the ratio of the test set,
# which we use to split up 20% of the data in to the test set and 80% for training.

from sklearn.tree import DecisionTreeClassifier
# criterion = 'gini' 'entropy'
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Evaluating the Algorithm
# At this point we have trained our algorithm and made some predictions.
# Now we'll see how accurate our algorithm is.
# For classification tasks some commonly used metrics are confusion matrix,
# precision, recall, and F1 score.
# Lucky for us Scikit=-Learn's metrics library contains the classification_report and confusion_matrix methods
# that can be used to calculate these metrics for us:

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
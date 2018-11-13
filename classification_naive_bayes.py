import pandas as pd
from sklearn.model_selection import train_test_split
import time
from sklearn.naive_bayes import GaussianNB


# url: https://blog.sicara.com/naive-bayes-classifier-sklearn-python-example-tips-42d100429e44
# url: https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/
# url; https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/

# # There are three types of Naive Bayes model under scikit learn library:
# #
# # Gaussian: It is used in classification and it assumes that features follow a normal distribution.
# #
# # Multinomial: It is used for discrete counts. For example, let’s say,  we have a text classification problem. Here we can consider bernoulli trials which is one step further and instead of “word occurring in the document”, we have “count how often word occurs in the document”, you can think of it as “number of times outcome number x_i is observed over the n trials”.
# #
# # Bernoulli: The binomial model is useful if your feature vectors are binary (i.e. zeros and ones). One application would be text classification with ‘bag of words’ model where the 1s & 0s are “word occurs in the document” and “word does not occur in the document” respectively.
#
# # we will be using gaussian library(normal/gaussian distribution)
#

# titanic dataset
import numpy as np

# Importing dataset
data = pd.read_csv("train.csv")

# Convert categorical variable to numeric
data["Sex_cleaned"]=np.where(data["Sex"]=="male",0,1)
data["Embarked_cleaned"]=np.where(data["Embarked"]=="S",0,np.where(data["Embarked"]=="C",1,np.where(data["Embarked"]=="Q",2,3)))
# Cleaning dataset of NaN
data=data[[
    "Survived",
    "Pclass",
    "Sex_cleaned",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_cleaned"
]].dropna(axis=0, how='any')

# Split dataset in training and test datasets
X_train, X_test = train_test_split(data, test_size=0.5, random_state=int(time.time()))

# Instantiate the classifier
gnb = GaussianNB()
used_features =[
    "Pclass",
    "Sex_cleaned",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_cleaned"
]

# Train classifier
gnb.fit(
    X_train[used_features].values,
    X_train["Survived"]
)
y_pred = gnb.predict(X_test[used_features])

# Print results
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])
))


# now another dataset

# df = pd.read_csv('weather_play.csv')
# # print(df)
# # print(df['Weather'])
# # print(df["Weather"])
#
# # convert categorical data into numeric data
#
# df["Weather_cleaned"] = np.where(df['Weather']=="Sunny",1,np.where(df["Weather"]=="Rainy",2,np.where(df["Weather"]=="Overcast",3,1)))
# print(df)
# df['Play_cleaned'] = np.where(df['Play'] == 'Yes',1,0)
# print(df)
# model = GaussianNB()
# model.fit(df[['Weather_cleaned']].values,df['Play_cleaned'])
#
# dict = {'Weather_cleaned':[1,2,3]}
# X_predict = pd.DataFrame([[1,2,3]])
# #print(X_predict.values)
#
# species_pred = model.predict(X_predict)
# print(species_pred)


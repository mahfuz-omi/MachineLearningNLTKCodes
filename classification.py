# classification by logistic regression
# url: http://www.data-mania.com/blog/logistic-regression-example-in-python/
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# rules for logistic regression:
# Target variable is binary(binary classification problems=problems with two class values)
# Predictive features are interval (continuous) or categorical
# Features are independent of one another
# Sample size is adequate – Rule of thumb: 50 records per predictor
# A key difference from linear regression is that the output value being modeled is a binary values (0 or 1) rather than a numeric value.

df = pd.read_csv('train.csv')
print(df)

print(list(df.columns))

# PRINT # OF NULL Values of each column
print(df.isnull().sum())

# print all columns info
print(df.info())

# drop irrelevant columns
df.drop(['PassengerId','Name','Ticket','Cabin','Embarked'],inplace=True,axis=1 )

print(df)

print(df.groupby('Pclass').mean())

# Survived        Age     SibSp     Parch       Fare
# Pclass
# 1       0.629630  38.233441  0.416667  0.356481  84.154687
# 2       0.472826  29.877630  0.402174  0.380435  20.662183
# 3       0.242363  25.140620  0.615071  0.393075  13.675550

# so, replace the null age with their mean age for the corresponding Pclass
#print(df[['Age','Pclass']])
# set age on null age place based on Pclass
def age_approx(columns):
    Age = columns[0]
    Pclass = columns[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

df['Age'] = df[['Age', 'Pclass']].apply(age_approx, axis=1)
print(df.isnull().sum())

# output:
# Survived    0
# Pclass      0
# Sex         0
# Age         0
# SibSp       0
# Parch       0
# Fare        0
# Embarked    2
# dtype: int64

# delete the 2 null values Embarked

df.dropna(inplace=True)
print(df)

# Converting categorical variables to a dummy indicators
# The next thing we need to do is reformat our variables so that they work with the model.
# Specifically, we need to reformat the Sex and Embarked variables into numeric variables.

gender = pd.get_dummies(df['Sex'],drop_first=True)
print(gender.head(5))

# embarked = pd.get_dummies(df['Embarked'],drop_first=True)
# print(embarked.head(5))

df.drop(['Sex'],axis=1,inplace=True)
print(df.head())

# concat 3 dataframe
titanic_dmy = pd.concat([df,gender],axis=1)
print(titanic_dmy.head())

# now check co-rellation between vars except survived
print(titanic_dmy.corr())

# fare and pclass are inter-related
# so drop these 2 columns
titanic_dmy.drop(['Fare', 'Pclass'],axis=1,inplace=True)
print(titanic_dmy.head())

# drop these 2 cols for no reason
titanic_dmy.drop(['SibSp', 'Parch'],axis=1,inplace=True)
print(titanic_dmy.head())

print(titanic_dmy.info())

# now regression
X = titanic_dmy.ix[:,(1,2)].values
print(X)
y = titanic_dmy.ix[:,0].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)
LogReg = LogisticRegression()

print('omi X train')
print(X_train)
print('omi y train')
print(y_train)
LogReg.fit(X_train, y_train)

y_pred = LogReg.predict(X_test)
# df1 = pd.DataFrame(list(y_pred))
# print(df1)
# df2 = pd.DataFrame(list(y_test))
# print(df2)
# df = pd.concat([df1,df2],axis=1)
# print(df)

# now check the success rate
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# output:
# [[140  25]
#  [ 36  67]]

# 140= test 0 pred 0,25=test 0 pred 1
# 36 = test 1 pred 0, 67= test 1 pred 1

# so accuracy is = 140+67 = 207 out of 268(140+67+25+36)
# accuracy rate = 77%

# print(titanic_dmy[['Age','male']].values)
# print(titanic_dmy['Survived'].values)
# we could write X_train = titanic_dmy[['Age','male']].values, y_train = titanic_dmy['Survived'].values













import pandas as pd

list = [1,3,5,7]
print(list)


series = pd.Series(list)
print(series)

print(series.index)

print(type(series.values))

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

# load dataset only the col_names column will be loaded
pima = pd.read_csv("pima-indians-diabetes.csv",names=col_names)


# convert caterorical data to numeric data (needed in logistic regression)
gender = pd.get_dummies(df['Sex'],drop_first=True)

df.drop(['Sex'],axis=1,inplace=True)

# concat 3 dataframe
titanic_dmy = pd.concat([df,gender],axis=1)
# df[sex]:
#
#
# index sex
# 0     male
# 1     female
#
# after dummies
# index male female
# 0       1      0
# 1       0      1
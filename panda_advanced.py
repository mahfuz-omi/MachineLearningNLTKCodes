import random

import pandas as pd

# url: https://dzone.com/articles/python-pandas-tutorial-series

myList = [1,3,5,7,9]
myTupple = (2,4,6,8,10)
myDict = {"name":'omi','age':25,'sex':'male'}

# any data structure, such as-list,tuple,dict can be converted to series
# index isn't set, so default index[0,1,2,3,4] is used
series1 = pd.Series(data=myTupple)

# series = series.index + series.values(1 dimensional)

print(series1)

# 0     2
# 1     4
# 2     6
# 3     8
# 4    10

# series1-series1.index
print(series1.values)
# out: [ 2  4  6  8 10]

print(series1.index.values)
# out: [0 1 2 3 4]

series2 = pd.Series(data=myList)
series3 = series1+series2
print(series3)

# each data item will be added
# 0     3
# 1     7
# 2    11
# 3    15
# 4    19

# addition is done based on index value

series2 = pd.Series(data=myList,index=[2,3,4,5,6])
series3 = series1+series2
print(series3)

# 0     NaN
# 1     NaN
# 2     7.0
# 3    11.0
# 4    15.0
# 5     NaN
# 6     NaN

# only the same index values are added
print(series3.isnull().sum())
# 4

# series3 isn't changed
series4 = series3.fillna(1)
print(series4)

# series3 has been changed
series3.fillna(1,inplace=True)
print(series4)

def function(data):
    print('omi  ',data)

# apply for all datas of series4
# send input as the data as data param in function
series4.apply(function)

#out:
# omi   1.0
# omi   1.0
# omi   7.0
# omi   11.0
# omi   15.0
# omi   1.0
# omi   1.0

print(series4.describe())
# count     7.000000
# mean      5.285714
# std       5.822780
# min       1.000000
# 25%       1.000000
# 50%       1.000000
# 75%       9.000000
# max      15.000000

# drop none value from row(axis=0), axis=1 means column value
series4.dropna(axis=0)

# series with name

series5 = pd.Series(data=[10,11,12,13,14],name='omi data')
print(series5)
# 0    10
# 1    11
# 2    12
# 3    13
# 4    14

df1 = pd.DataFrame(data=series5)
print(df1)
#         omi data
# 0        10
# 1        11
# 2        12
# 3        13
# 4        14

# create series with scaler data
series6 = pd.Series(2, index=range(0, 5))
print(series6)

# 0    2
# 1    2
# 2    2
# 3    2
# 4    2
# dtype: int64

series7 = pd.Series(data=range(0,10))
print(series7)
# 0    0
# 1    1
# 2    2
# 3    3
# 4    4
# 5    5
# 6    6
# 7    7
# 8    8
# 9    9
# dtype: int64

# alphabetical index
# num of element is 5
series8 = pd.Series(data=range(1, 15, 3), index=[x for x in 'abcde'])
print(series8)

# a     1
# b     4
# c     7
# d    10
# e    13
# dtype: int64

# slicing series
print(series8[1])
# 4

print(series8[0:3])
# a    1
# b    4
# c    7
# dtype: int64

# random numbers
# here, the random_data is a list
random_num = random.sample(range(0,100),10)
print(random_num)
#[68, 11, 32, 48, 0, 64, 40, 95, 60, 47]

# conditional series data
print(series8[series8>4])
# c     7
# d    10
# e    13

# indexing series data
print(series8[[0,1,4]])
# a     1
# b     4
# e    13


# data frame
print(df1)

# create df from dict
# all values for a key is a list
# this is column oriented dataframe creation from dict

dict1 = {'jan':[10,20,30,35],'feb':[15,20,18,27],'mar':[25,20,15,12]}
df2 = pd.DataFrame(dict1)

print(df2)
#     feb  jan  mar
# 0   15   10   25
# 1   20   20   20
# 2   18   30   15
# 3   27   35   12

# set index
df2 = pd.DataFrame(dict1,index=[3,5,7,9])
print(df2)

#     feb  jan  mar
# 3   15   10   25
# 5   20   20   20
# 7   18   30   15
# 9   27   35   12

print(type(df2['jan']))
# it's a series

print(df2['jan'])
# 3    10
# 5    20
# 7    30
# 9    35
# Name: jan, dtype: int64

print(df2['jan'].values)
# [10 20 30 35]

print(df2.values)
# [[15 10 25]
#  [20 20 20]
#  [18 30 15]
#  [27 35 12]]

# cropping df
# first argument:row selection,second argument:column selection
print(df2.values[:,0:1])
# [[15]
#  [20]
#  [18]
#  [27]]

# first row
print(df2.values[0])
# [15 10 25]

# first row,first column
print(df2.values[0][0])
# 15

# first row,first column
print(df2.values[0,0])
# 15

# first row,first column(integer index)
print(df2.iloc[0][0])
# 15

# first row,first column, with actual index, 3 for first row index, 'feb' is for column index
print(df2.loc[3]['feb'])
# 15
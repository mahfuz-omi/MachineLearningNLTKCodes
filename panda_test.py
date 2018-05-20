#python list


# create an empty list with []
list = []
for i in range(10):
    list.append(i**2)

print(list)

fruits = ['orange', 'apple', 'pear', 'banana', 'kiwi', 'apple', 'banana']
print(len(fruits))
print(fruits.index('banana'))
fruits.reverse()
print(fruits)
print(fruits.count('banana'))
fruits.append('grape')
print(fruits)
fruits.sort()
print(fruits)
fruits.insert(1,'omi')
print(fruits)

# remove only first appearance of this object
fruits.remove('apple')
print(fruits)

# create an empty set with set()
# no duplicate element, and nor ordering among elements
# set = {} will create an empty dict
set = set()
set.add(3)
set.add(5)
set.add(7)
set.add(5)
print(set)

set1 = {4,6,4,7,6,9}
print(set1)
print(len(set1))
print(4 in set1)

# diff between json and python dict
# json only has string as keys, whereas dict may have other objects as keys

# [] creates an empty dict, not set
dict = {}
dict['omi'] = 104
dict['akash'] = 73
dict['ratul'] = 43

dict2 = {'a':12,'b':17,'c':20}
print(dict2)
dict2['d'] = 12

print(dict['omi'])

print(dict2)

# deleting dict
#dict.clear()     # remove all entries in dict
#del dict


print(dict.keys())
keys = dict.keys();


# pandas series
import pandas as pd
series1 = pd.Series([1,3,5,7],index=[1,2,3,4])
# the index isn't a part of the data
# so, the data is one dimensional
print(series1)
print(series1[4])


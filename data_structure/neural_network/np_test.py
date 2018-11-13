# https://www.guru99.com/numpy-tutorial.html
import numpy as np
myPythonList = [[1,9,8,3],[2,2,2,3]]
np_array = np.array(myPythonList)

print(np_array)
# [1 9 8 3]
# the commas are eliminated

print(type(np_array))
# <class 'numpy.ndarray'>

# array is a function to create ndarray

print(np_array.dtype)
# int32

# type() returns the parent data type, dtype returns the children data type

print(np_array.shape)

# total number of single element
print(np_array.size)

print(np_array.reshape(4,2))

# resize change the array itself(like inPlace = True)
# np_array.resize(4,2)

# though reshape, np_array isn't changed
print(np_array)

# dimension reduction, convert to 1-D
print(np_array.flatten())


#Shape: is the shape of the array
#Dtype: is the datatype. It is optional. The default value is float64
print(np.zeros((2,2),dtype=np.int16))


# + - * / operates for each element in ndarray
print(np_array+3)
# [[ 4 12 11  6]
#  [ 5  5  5  6]]

# Numpy library has also two convenient function
# to horizontally or vertically append the data.
# Lets study them with an example:

f = np.array([1,2,3])
g = np.array([4,5,6])

print('Horizontal Append:', np.hstack((f, g)))
print('Vertical Append:', np.vstack((f, g)))


# Horizontal Append: [1 2 3 4 5 6]
# Vertical Append: [[1 2 3]
#                   [4 5 6]]

# create array with arange
print(np.arange(1,11))
# [ 1  2  3  4  5  6  7  8  9 10]

print(np.arange(1,11,2))
# [1 3 5 7 9]


# slicing numpy
e  = np.array([(1,2,3), (4,5,6)])
print(e)
# [[1 2 3]
#  [4 5 6]]

# first row
print(e[0])

# second row
print(e[1])

# first column (all rows but only 1st column)
print(e[:,0])

# second column (all rows but only 2nd column)
print(e[:,1])

#return the first two values of the second row.
print(e[1,:2])

# Min
#
# np.min()
#
# Max
#
# np.max()
#
# Mean
#
# np.mean()
#
# Median
#
# np.median()
#
# Standard deviation
#
# np.stdt()

# Dot Product
# Numpy is powerful library for matrices computation.
# For instance, you can compute the dot product with np.dot

## Linear algebra
### Dot product: product of two arrays
f = np.array([1,2])
g = np.array([4,5])
### 1*4+2*5
print(np.dot(f, g))
# 14

# loop
# item is a list here
# to loop through all single item, inner loops are required
print("np array",np_array)
for item in np_array:
    print("item",item)

# np array [[1 9 8 3]
#           [2 2 2 3]]
# item [1 9 8 3]
# item [2 2 2 3]

# random numbers between 0-1
print(np.random.rand())

# random numbers of shape (2,3)
print(np.random.rand(2,3))


import numpy as np

# numpy array supports indexing as well as slicing

# 1d array
array = np.array([1,3,5,7])
print(array)

print(array.ndim)
print(array.shape)
print(array[0])

# [1 3 5 7]
# 1
# (4,)
# 1

# 2d array
array = np.array([ [1,3,5],[2,4,6] ])

print(array)

print(array.ndim)
print(array.shape)
print(array[0])
print(array[0][1])
print(array.flatten())
print(array[0:1,0:1])



# [[1 3 5]
#  [2 4 6]]
# 2
# (2, 3)
# [1 3 5]
# 3
# [1 3 5 2 4 6]
# [[1]]

# numpy array can be reshaped

print(array.reshape(3,2))

# [[1 3]
#  [5 2]
#  [4 6]]

# convert 2d to 3d
print(array.reshape(2,3,1))

# [[[1]
#   [3]
#   [5]]
#
#  [[2]
#   [4]
#   [6]]]


# 3d array
array = np.array([

        [[1,2,3,7],[3,4,5,8],[2,8,9,9]],[[5,6,7,0],[7,8,9,1],[5,5,6,0]]

    ])

print(array)

print(array.ndim)
print(array.shape)
print(array.flatten())

# [[[1 2 3 7]
#   [3 4 5 8]
#   [2 8 9 9]]
#
#  [[5 6 7 0]
#   [7 8 9 1]
#   [5 5 6 0]]]
# 3
# (2, 3, 4)
# [1 2 3 7 3 4 5 8 2 8 9 9 5 6 7 0 7 8 9 1 5 5 6 0]

# flatten always convert to 1D

# if X is 3d, then [X] is 4d
# if shape of X is a,b,c then shape of [x] is 1,a,b,c


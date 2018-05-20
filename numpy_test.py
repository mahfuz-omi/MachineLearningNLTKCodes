import numpy as np
a = np.array([1,2,3])
print(a)
print(a.dtype)

b = np.array([1.2, 3.5, 5.1])
print(b.dtype)

c = np.arange(0,11)
print(c)

d = np.arange(12)
print(d)

e = d.reshape(4,3)
print(e)

# works for each elements of numpy array
print(2*e+4)

# reshaping

data = [[11, 22],
		[33, 44],
		[55, 66]]
# array of data
data = np.array(data)
print(data.shape)
print('Rows: %d' % data.shape[0])
print('Cols: %d' % data.shape[1])

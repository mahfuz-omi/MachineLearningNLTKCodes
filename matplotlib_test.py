import matplotlib.pyplot as plt
import numpy as np

# plot with list data
# only y value is provided here, default value for x i s[0,1,2....]
# f you provide a single list or array to the plot() command, matplotlib assumes it is a sequence of y values
#
# and automatically generates x values for you
#
# the default x vector has same length as y but start with 0
# which is the format string
#
# indicates the color
#
# and line type of the plot
#
# the letters and symbols of the format string are from MATLAB
#
# default format string is b-, which is a solid blue line
y = [0,1,4,9,16]
x = [0,1,2,3,4]
plt.plot(y)
plt.show()


# plot with red circle
plt.plot(x,y,'ro')
plt.show()


# x and y values are provided here
plt.plot(x,y)
plt.show()

# plot with numpy data(array)
y_np = np.arange(20)
x_np = np.arange(20)
plt.title('numpy data')
plt.ylabel('Y axis')
plt.xlabel('X axis')


# plot multiple datas here, with diff color.
# show means flush
# after flushing, datas are inserted here newly
plt.plot(x_np,y_np,linewidth=5)
plt.plot(x,y,linewidth=5)
plt.show()

# using legend for multiple data
plt.plot(x_np,y_np,linewidth=5,label="first line")
plt.plot(x,y,linewidth=5,label="second line")
plt.legend()
plt.show()

# bar chart
plt.bar(x,y)
plt.bar(x_np,y_np)
plt.legend()
plt.show()


# scatter plot
plt.scatter(x,y)
plt.scatter(x_np,y_np)
plt.show()

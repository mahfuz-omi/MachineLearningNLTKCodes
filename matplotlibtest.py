import matplotlib.pyplot as plt
import pandas_test as pd
#plt.plot([1,2,3,4,5])

data = pd.read_csv('data.csv')
print(data)
# plt.plot(data)
# plt.show()
print(data['name'])
print(data['age'])
print(data['id'])
x = data['id']
y = data['age']
plt.plot(x,y,'ro')
plt.show()


# default id b- means blue line
# ro means red o
# bo means blue o
# plt.plot([1, 2, 3, 4], [1, 4, 9, 16],'bo')
# plt.xlabel('this is x')
# plt.ylabel('this is y')
# # xmin, xmax, ymin, ymax
# plt.axis([0,6,0,20])
# plt.show()


# plt.plot([1, 2, 3, 4], [1, 4, 9, 16],linewidth=10.0)
# plt.show()
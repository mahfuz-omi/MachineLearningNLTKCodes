import pandas_test as pd
import matplotlib.pyplot as plt
import scipy
data = pd.read_csv('AirPassengers.csv')
#print(data.head(10))
# print('\n Data Types:')
# #print(data.dtypes)
#
# dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
# data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)
#print(data.head(100))



dict = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
       "area": [8.516, 17.10, 3.286, 9.597, 1.221],
       "population": [200.4, 143.5, 1252, 1357, 52.98] }
#
# print(dict)
#
# import pandas as pd
brics = pd.DataFrame(dict,index=['omi','popy','sumi','arrna','ma'])
# # print(brics)
#
# # Set the index for brics
# #brics.index = ["BR", "RU", "IN", "CH", "SA"]
#
# # Print out brics with new index values
# print(brics)
#
# data = pd.read_csv('data.csv',index_col=0)
# print(data)
#
# print(data['name'])
# print(data['age'])
#
# # data[a:b] : from a(inclusive) upto b(exclusive)
# # data[a:]
# # data[:b]
# print(data[2:3])
#
# # plot data
# import matplotlibtest
# import matplotlibtest.pylab as plt
#plt.plot([1,2,3])

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)
#print(data.head())
#print(data.index)

ts = data['#Passengers']
#ts.head(10)
#print(data['1949'])

#plt.plot(ts)
#plt.show()

print(ts)

from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries):
       # Determing rolling statistics
       rolmean = pd.rolling_mean(timeseries, window=12)
       rolstd = pd.rolling_std(timeseries, window=12)

       # Plot rolling statistics:
       orig = plt.plot(timeseries, color='blue', label='Original')
       mean = plt.plot(rolmean, color='red', label='Rolling Mean')
       std = plt.plot(rolstd, color='black', label='Rolling Std')
       plt.legend(loc='best')
       plt.title('Rolling Mean & Standard Deviation')
       plt.show(block=False)

       # Perform Dickey-Fuller test:
       print('Results of Dickey-Fuller Test:')

       dftest = adfuller(timeseries, autolag='AIC')
       dfoutput = pd.Series(dftest[0:4],
                            index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
       for key, value in dftest[4].items():
              dfoutput['Critical Value (%s)' % key] = value
       print(dfoutput)

test_stationarity(ts)

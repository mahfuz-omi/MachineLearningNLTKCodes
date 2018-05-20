import pandas as pd
import matplotlib.pyplot as plt
import patsy

#df = pd.read_csv('birds.csv')
#print(df.head(10))
#ts = df['Daily total female births in California, 1959']
#print(ts)

# plt.plot(ts[0:365])
# plt.show()
# plt.hist(ts[0:365])
# plt.show()

# check whether is's stationary or not
# method = summary statistics
# we can split the time series into two contiguous sequences.
# We can then calculate the mean and variance of each group of numbers and compare the values.

# take inly values, not index
# X = ts[0:365].values
# #print(X)
# split = len(X) / 2
# split = int(split)
# X1, X2 = X[0:split], X[split:]
# mean1, mean2 = X1.mean(), X2.mean()
# var1, var2 = X1.var(), X2.var()
# print('mean1=%f, mean2=%f' % (mean1, mean2))
# print('variance1=%f, variance2=%f' % (var1, var2))


# output=
#mean1=39.763736, mean2=44.185792
#variance1=49.213410, variance2=48.708651

# they are very much closer
# so, the series is stationary

# test another non-stationary dataset
# df2 = pd.read_csv('AirPassengers.csv')
# ts2 = df2['#Passengers']
# print(ts2)
#
# plt.plot(ts2)
# plt.show()
# value = ts2.values;
# split2 = len(value) / 2
# split2 = int(split2)
# X1, X2 = value[0:split2],value[split2:]
#
# mean, mean2 = X1.mean(), X2.mean()
# var1, var2 = X1.var(), X2.var()
# print('mean1=%f, mean2=%f' % (mean1, mean2))
# print('variance1=%f, variance2=%f' % (var1, var2))


#output=
#mean1=39.763736, mean2=377.694444
#variance1=2244.087770, variance2=7367.962191

# so huge difference between them
# so this time series isn't stationary

# now Augmented Dickey-Fuller test
# need to import patsy
from statsmodels.tsa.stattools import adfuller

# df = pd.read_csv('birds.csv')
# ts = df['Daily total female births in California, 1959']
# plt.plot(ts[0:365])
# plt.show()
# X = ts[0:365].values
# result = adfuller(X)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# print('Critical Values:')
# for key, value in result[4].items():
# 	print('\t%s: %.3f' % (key, value))

# if p value > 0.05 = non-stationary
# if p value <= 0.05 = stationary
# output=
# ADF Statistic: -4.808291
# p-value: 0.000052
# Critical Values:
# 	1%: -3.449
# 	5%: -2.870
# 	10%: -2.571

# this model is stationary.
# if not-stationary, we would need to make it stationary
# apply log function on time series

# first consider a non-stationary series
# df = pd.read_csv('AirPassengers.csv')
# ts = df['#Passengers']
# #plt.plot(ts)
# #plt.show()
# X = ts.values
# result = adfuller(X)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# print('Critical Values:')
# for key, value in result[4].items():
# 	print('\t%s: %.3f' % (key, value))

# output=
	# ADF
	# Statistic: 0.815369
	# p - value: 0.991880
	# Critical
	# Values:
	# 1 %: -3.482
	# 5 %: -2.884
	# 10 %: -2.579

# so, this model is non-stationary
# so make it stationary
import numpy as np
# ts_log = np.log(ts)
#plt.plot(ts)

# X = ts.values
# result = adfuller(X)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# print('Critical Values:')
# for key, value in result[4].items():
# 	print('\t%s: %.3f' % (key, value))

# output =
# ADF Statistic: -1.717017
# p-value: 0.422367
# Critical Values:
# 	1%: -3.482
# 	5%: -2.884
# 	10%: -2.579

# it's still non-stationary
# apply something more

# smoothing techniques
# rolling mean

# moving_avg = pd.rolling_mean(ts_log,12)
# plt.plot(ts_log)
# plt.plot(moving_avg, color='red')
# plt.show()
# print(moving_avg.head())

# ts_diff = ts_log - moving_avg
# plt.plot(ts_diff)
# plt.show()

# drop the first 11 datas as they are Nan
# ts_diff.dropna(inplace=True)
#plt.plot(ts_diff)
#plt.show()

# X = ts_diff.values
# result = adfuller(X)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# print('Critical Values:')
# for key, value in result[4].items():
# 	print('\t%s: %.3f' % (key, value))

# output=
# ADF Statistic: -3.162908
# p-value: 0.022235
# Critical Values:
# 	1%: -3.487
# 	5%: -2.886
# 	10%: -2.580

# now it's turn out to be a stationary data
# but, we could expect better approach
# weighted moving average rather than only moving/rolling average

# ewma = exponentially weighted moving average
# ewma = pd.ewma(ts_log, halflife=12)
# plt.plot(ts_log)
# plt.plot(ewma, color='red')
# plt.show()
#
# ts_diff = ts_log-ewma
# plt.plot(ts_diff)
# plt.show()
#
# X = ts_diff.values
# result = adfuller(X)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# print('Critical Values:')
# for key, value in result[4].items():
# 	print('\t%s: %.3f' % (key, value))

# there are more techniques for removing trend
# now we will use differentiate technique
# Detrend by Differencing

#df = pd.read_csv('shampoo.csv')
#print(df)

#series = df['Sales of shampoo over a three year period']
# plt.plot(series)
# plt.show()
# print(series)
# X = series.values
# print("X: ",X)
# diff = list()
# for i in range(1, len(X)):
# 	value = X[i] - X[i - 1]
# 	diff.append(value)
# plt.plot(diff)
# plt.show()

# series = pd.Series(diff)
# # remove Nan values from series
# series.dropna(inplace=True)
# X = series.values
#
#
# result = adfuller(X)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# print('Critical Values:')
# for key, value in result[4].items():
# 	print('\t%s: %.3f' % (key, value))

# output:
# ADF Statistic: -7.249074
# p-value: 0.000000
# Critical Values:
# 	1%: -3.646
# 	5%: -2.954
# 	10%: -2.616

# so, this model is stationary now

# Detrend by Model Fitting
# regression technique

# fit linear model

# from sklearn.linear_model import LinearRegression
# series.dropna(inplace=True)
# X = [i for i in range(0, len(series))]
# print("X",X)
# X = np.reshape(X, (len(X), 1))
# print("x",X)
# y = series.values
# model = LinearRegression()
# model.fit(X, y)
# # calculate trend
# trend = model.predict(X)
# # plot trend
# plt.plot(y)
# plt.plot(trend)
# plt.title("model fitting")
# plt.show()
# # # detrend
# detrended = [y[i]-trend[i] for i in range(0, len(series))]
# # plot detrended
# plt.plot(detrended)
# plt.show()

# now decomposition trend seasonal and residual
from statsmodels.tsa.seasonal import seasonal_decompose
df = pd.read_csv('AirPassengers.csv')
# print(df)
# print(df['#Passengers'])
# print(df['#Passengers'].values)

series = df['#Passengers']
#plt.plot(series)
result = seasonal_decompose(series,freq=5)
result.plot()
plt.show()

residul = result.resid
residul.dropna(inplace=True)
print(residul)

plt.plot(residul)
plt.show()

X = residul.values
# now check stationary
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# output=
# ADF Statistic: -7.009681
# p-value: 0.000000
# Critical Values:
# 	1%: -3.483
# 	5%: -2.885
# 	10%: -2.579

# so, the residual is stationary(trend and seasonality have been removed)

# now time to forecast data













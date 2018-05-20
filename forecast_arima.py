import warnings

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt
import patsy
import numpy as np
from statsmodels.tsa.stattools import adfuller

# dickey-fuller test
def test_stationary(ts):
    X = ts.values
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


# input data
df = pd.read_csv('AirPassengers.csv')
series = df['#Passengers']
series.plot()
plt.title('raw data')
plt.show()

test_stationary(series)

# seasonal decomposition to make it stationary
# result = seasonal_decompose(series,freq=5)
# result.plot()
# plt.title('seasonal decompose')
# plt.show()
#
# decomposed = result.resid
# decomposed.dropna(inplace=True)
# plt.plot(decomposed)
# plt.title('decomposed residual')
# plt.show()
#
# # test stationary of decomposed residual
# test_stationary(decomposed)


# log series data
# series_log = np.log(series)
# plt.plot(series_log)
# plt.show()

# now we will go with series_log

# make this data stationary by difference method
# series_log_diff = series_log - series_log.shift()
# plt.plot(series_log_diff)
# plt.show()

# now check whether this data is stationary or not
# series_log_diff.dropna(inplace=True)
# test_stationary(series_log_diff)

# the data has become stationary by first order difference. So, d = 1
# now calculate p and q
# 2 systems.
# 1. acf and pacf function
# 2. grid-search
# we will use grid search method

# ARIMA model
# from statsmodels.tsa.arima_model import ARIMA
#
#
# # evaluate an ARIMA model for a given order (p,d,q)
from sklearn.metrics import mean_squared_error
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error
#
#
# # evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order, mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
#
#
# # evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series, p_values, d_values, q_values)
#
#
# # (p,d,q) = (2,1,1) where min square error is minimum
# model = ARIMA(series_log, order=(2, 1, 2))
# results_ARIMA = model.fit(disp=-1)
# plt.plot(series_log_diff)
# plt.plot(results_ARIMA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-series_log_diff)**2))
#

# differencing
series_diff = series-series.shift()
series_diff.dropna(inplace=True)
plt.plot(series_diff)
plt.show()

test_stationary(series_diff)

# arima model
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order, mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


def plot_arima(X):
    #prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    #make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(8,1,0))
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    #calculate out of sample error
    error = mean_squared_error(test, predictions)
    plt.plot(test)
    plt.plot(predictions)
    plt.show()
    return error


# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
#evaluate_models(series.values, p_values, d_values, q_values)

# So, (8,1,0) i the lowest mse providing order
# model = ARIMA(series, order=(8,1,0))
# model_fit = model.fit(disp=0)
# #one-step out-of sample forecast
# forecast = model_fit.forecast()
# print(forecast)
# data = forecast[0]
# print("data:",data)

error = plot_arima(series.values)
print(error)

# predict bulk data
# mode.predict(startIndex, endIndex)










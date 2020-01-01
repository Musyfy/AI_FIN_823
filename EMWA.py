import quandl
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

######################
quandl.ApiConfig.api_key = "uRMo697HgMj91ZZZa2_v"
data = quandl.get_table('WIKI/PRICES', qopts = { 'columns': ['ticker', 'date', 'adj_close'] }, ticker = ['AAPL'], date = { 'gte': '2009-01-01', 'lte': '2019-12-19' })
data.sort_values(by='date', ascending=True,inplace =True)
data.set_index(pd.DatetimeIndex(data['date']),inplace=True)
data = data[['adj_close']]
plt.plot(data)




def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(2).mean()
    rolstd = timeseries.rolling(2).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print
    'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries['adj_close'].values, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print
    dfoutput


test_stationarity(data)

data2= np.log(data)
plt.plot(data2)

expwighted_avg = pd.Series.ewm(data2['adj_close'], span=2).mean()
plt.plot(data2)
plt.plot(expwighted_avg, color='red')

#data2_log_ewma_diff = data2 - expwighted_avg
#test_stationarity(data2_log_ewma_diff)

data2_diff = data2 - data2.shift()
plt.plot(data2_diff)
data2_diff.dropna(inplace=True)
test_stationarity(data2_diff)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data2['adj_close'].values, freq=30)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(data2, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

#data2_decompose = residual
#data2_decompose.dropna(inplace=True)
#test_stationarity(data2_decompose)

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(data2['adj_close'].values, order=(5, 1, 2))
results_ARIMA = model.fit(disp=1)
plt.plot(data2_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-data2_diff)**2))

from arch import arch_model
from random import gauss
from random import seed
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

# seed pseudorandom number generator
seed(1)
# split into train/test
n_test = 10
data3 = sc.fit_transform(data2_diff)
train, test = data3[:-n_test], data3[-n_test:]
# define model
model = arch_model(train, vol='GARCH', p=1, q=1)
# fit model
model_fit = model.fit()
print(model_fit.summary())
# forecast the test set
yhat = model_fit.forecast(horizon=n_test)
# plot the actual variance
var = [i*0.01 for i in range(0,100)]
plt.plot(var[-n_test:])
# plot forecast variance
plt.plot(yhat.variance.values[-1, :])
plt.show()

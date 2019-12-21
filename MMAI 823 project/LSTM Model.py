import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
######################
quandl.ApiConfig.api_key = "uRMo697HgMj91ZZZa2_v"
data = quandl.get_table('WIKI/PRICES', qopts = { 'columns': ['ticker', 'date', 'adj_close'] }, ticker = ['AAPL'], date = { 'gte': '2009-01-01', 'lte': '2019-12-19' })
data.sort_values(by='date', ascending=True,inplace =True)
data_1 = data.iloc[:1957]
data_2 = data[1957:]###365
plt.figure(figsize = (20,14))
plt.plot(range(data.shape[0]),(data['adj_close']))
plt.xticks(range(0,data.shape[0],1000),data['date'].loc[::1000],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Closed Price',fontsize=18)
plt.show()
#######################
training_set = data_1.iloc[:,2:3].values
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
X = []
y = []
for i in range(30, len(training_set_scaled)):
    X.append(training_set_scaled[i-30:i, 0])
    y.append(training_set_scaled[i, 0])
X, y = np.array(X), np.array(y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


regressor = Sequential()

regressor.add(LSTM(units = 60, activation = "relu",return_sequences = True, input_shape = (X.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 80,  activation = "relu",return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 120,  activation = "relu"))
regressor.add(Dropout(0.4))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X, y, epochs = 100, batch_size = 32)


real_stock_price = data_2.iloc[:, 2:3].values

dataset_total = data['adj_close']
inputs = dataset_total[len(dataset_total) - len(data_2) -30:].values
inputs = inputs.reshape(-1,1)
inputs = sc.fit_transform(inputs)
X_test = []
for i in range(30, len(inputs)):
    X_test.append(inputs[i-30:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


plt.plot(real_stock_price, color = 'black', label = 'AAPL Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted AAPL Stock Price')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('AAPL Stock Price')
plt.legend()
plt.show()

# summarize history for loss
#plt.plot(regressor.history['loss'])
#plt.plot(regressor.history['val_loss'])
#plt.title('model loss')
##plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
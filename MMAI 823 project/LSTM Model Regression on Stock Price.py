import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.client import device_lib
from keras import backend as K
from keras import metrics
import keras as ks
import pickle
import pandas_datareader.data as pd_reader

device_lib.list_local_devices()
K.tensorflow_backend._get_available_gpus()

######################
#quandl.ApiConfig.api_key = "uRMo697HgMj91ZZZa2_v"
#data = quandl.get_table('WIKI/PRICES', qopts = { 'columns': ['ticker', 'date', 'adj_close','open','high','low','close'] }, ticker = ['AAPL'], date = { 'gte': '2009-01-01', 'lte': '2019-12-19' })
data = pd_reader.DataReader(
    'AAPL',
    'yahoo'
)
data = data[['Adj Close','High','Open','Close','Low']]
#data.sort_values(by='Date', ascending=True,inplace =True)
data = data.dropna()
factor_ratio = 0.8
data_1 = data.iloc[:round(len(data)*factor_ratio)]
data_2 = data[round(len(data)*factor_ratio):]

###############################################
##  Feature engineering construct the retrun ##
###############################################
training_set = data_1.iloc[:,0:].values
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


X = []
y = []
for i in range(45, len(training_set_scaled)):
    X.append(training_set_scaled[i-45:i])
    y.append(training_set_scaled[i,0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 5))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

hidden_layer_sizes = [200,300,300,200,100]
def build_model(hidden_layer_sizes,act,input_d):
  model = Sequential()

  model.add(LSTM(units = hidden_layer_sizes[0], activation= act,return_sequences= True, input_shape =input_d))
  model.add(Dropout(0.2))

  for layer_size in hidden_layer_sizes[1:len(hidden_layer_sizes)-1]:
    model.add(LSTM(units =layer_size, activation = act, return_sequences=True))
    model.add(Dropout(0.3))

  model.add(LSTM(units=hidden_layer_sizes[len(hidden_layer_sizes)-1], activation=act))
  model.add(Dropout(0.2))

  model.add(Dense(units=25))
  model.add(Dense(units=1,activation=act))

  model.compile(optimizer='adam', loss='mean_squared_error')
  return model




regressor = build_model(hidden_layer_sizes,"relu",(X.shape[1],5))


#callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             #ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

model = regressor.fit(X, y, epochs = 200, batch_size = 100)
plt.plot(model.history['loss'])
plt.show()

###

real_stock_price = data_2.iloc[:,0:1].values

dataset_total = data.iloc[:,0:]
inputs = dataset_total[len(dataset_total) - len(data_2) -45:].values
#inputs = inputs.reshape(-1,1)
inputs = sc.fit_transform(inputs)

X_test = []
for i in range(45, len(inputs)):
    X_test.append(inputs[i-45:i])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))
predicted_stock_price= regressor.predict(X_test)
sc.fit_transform(real_stock_price)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_stock_price = pd.DataFrame(predicted_stock_price,columns=['adj_close'])
predicted_stock_price.reset_index(inplace=True)
predicted_stock_price = predicted_stock_price[['adj_close']]
data_test = data[round(len(data)*factor_ratio):]
data_test.reset_index(inplace=True)
data_test = data_test[["Date"]]

final = pd.merge(data_test, predicted_stock_price, left_index=True, right_index=True)
#predicted_stock_price['return1'] = predicted_stock_price['close'].shift(1) / predicted_stock_price['close'] - 1

#predicted_stock_price['cum'] = (1 + predicted_stock_price.return1).cumprod() - 1


#d = predicted_stock_price["cum"].values

plt.plot(real_stock_price, color = 'black', label = 'AAPL Stock Price')
plt.plot(predicted_stock_price['adj_close'], color = 'green', label = 'Predicted AAPL Stock Price')
plt.title('AAPL Stock return Prediction')
plt.xlabel('Time')
plt.ylabel('AAPL Stock return')
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

import pickle
with open('Apple_model_regression.pickle','wb') as f:
    pickle.dump(regressor,f)

with open('Apple_model_regression.pickle','rb') as f:
    model = pickle.load(f)


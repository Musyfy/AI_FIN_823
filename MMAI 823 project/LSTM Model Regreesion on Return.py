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
###define treshold####
######################
tresh_hold = 0.002
#######################
###Lable function #####
#######################
def lable(x,tresh_hold):
    if x >= tresh_hold:
        return 1
    else:
        return 0

######################
#quandl.ApiConfig.api_key = "uRMo697HgMj91ZZZa2_v"
#data = quandl.get_table('WIKI/PRICES', qopts = { 'columns': ['ticker', 'date', 'adj_close','open','high','low','close'] }, ticker = ['AAPL'], date = { 'gte': '2009-01-01', 'lte': '2019-12-19' })
data = pd_reader.DataReader(
    'AAPL',
    'yahoo'
)
data = data[['Adj Close','High','Open','Close','Low']]
#data.sort_values(by='Date', ascending=True,inplace =True)
data["return"] = np.log(data["Adj Close"].shift(1))-np.log(data["Adj Close"])
#data["movement"] = data["return"].apply(lambda x:lable(x,tresh_hold))
data = data.dropna()
#data.drop(['return'],axis=1,inplace = True)
factor_ratio = 0.8
data_1 = data.iloc[:round(len(data)*factor_ratio)]
data_2 = data[round(len(data)*factor_ratio):]

###############################################
##  Feature engineering construct the retrun ##
###############################################
training_set = data_1.iloc[:,0:-1].values
y_set = data_1["return"].values
y_set = y_set.reshape(-1,1)
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
y_set_scaled = sc.fit_transform(y_set)


X = []
y = []
for i in range(252, len(training_set_scaled)):
    X.append(training_set_scaled[i-252:i])
    y.append(y_set_scaled[i][0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 5))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

hidden_layer_sizes = [100,100]
def build_model(hidden_layer_sizes,act,input_d):
  model = Sequential()

  model.add(LSTM(units = hidden_layer_sizes[0],go_backwards= True,return_sequences= True,input_shape =input_d))
  model.add(Dropout(0.5))

  model.add(LSTM(units=hidden_layer_sizes[len(hidden_layer_sizes)-1]))
  model.add(Dropout(0.5))

  model.add(Dense(units=25))
  model.add(Dense(units=1))

  model.compile(optimizer='adam', loss='mean_squared_error')
  return model




regressor = build_model(hidden_layer_sizes,"relu",(X.shape[1],5))


#callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             #ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

model = regressor.fit(X, y, epochs = 100, batch_size = 100)
plt.plot(model.history['loss'])
plt.show()

###

real_stock_price = data_2['return'].values

dataset_total = data.iloc[:,0:-1]
inputs = dataset_total[len(dataset_total) - len(data_2) -252:].values
#inputs = inputs.reshape(-1,1)
inputs = sc.fit_transform(inputs)

X_test = []
for i in range(252, len(inputs)):
    X_test.append(inputs[i-252:i])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))
predicted_stock_price= regressor.predict(X_test)
y_set_scaled = sc.fit_transform(y_set)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_stock = pd.DataFrame(predicted_stock_price,columns=['perdict return'])
real_stock = pd.DataFrame(real_stock_price,columns=['real return'])

final = pd.merge(real_stock, predicted_stock, left_index=True, right_index=True)
final['direction'] = final["real return"]*final["perdict return"]
def iden(x):
    if x >0:
        return 1
    else:
        return 0
final["hit"] = final["direction"].apply(iden)
print('hit ratio:{}'.format(final["hit"].sum()/final["hit"].count()))

plt.plot(real_stock_price, color = 'black', label = 'AAPL Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted AAPL Stock return')
plt.title('AAPL Stock return Prediction')
plt.xlabel('Time')
plt.ylabel('AAPL Stock return')
plt.legend()
plt.show()

import pickle
with open('Apple_model_regression_return.pickle','wb') as f:
    pickle.dump(regressor,f)

with open('Apple_model_regression_return.pickle','rb') as f:
    model = pickle.load(f)


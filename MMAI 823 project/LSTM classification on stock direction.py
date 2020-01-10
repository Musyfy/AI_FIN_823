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
from sklearn.metrics import accuracy_score, precision_score , recall_score, f1_score, confusion_matrix,cohen_kappa_score,log_loss

device_lib.list_local_devices()
K.tensorflow_backend._get_available_gpus()
######################
###define treshold####
######################
tresh_hold = 0
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
    'TSLA',
    'yahoo'
)
data = data[['Adj Close','High','Open','Close','Low']]
#data.sort_values(by='Date', ascending=True,inplace =True)
data["return"] = np.log(data["Adj Close"].shift(1))-np.log(data["Adj Close"])
data["movement"] = data["return"].apply(lambda x:lable(x,tresh_hold))
data = data.dropna()
data.drop(['return'],axis=1,inplace = True)
factor_ratio = 0.8
data_1 = data.iloc[:round(len(data)*factor_ratio)]
data_2 = data[round(len(data)*factor_ratio):]

###############################################
##  Feature engineering construct the retrun ##
###############################################
training_set = data_1.iloc[:,0:-1].values
y_set = data_1["movement"].values
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
#y_set_scaled = sc.fit_transform(y_set)


X = []
y = []
for i in range(30, len(training_set_scaled)):
    X.append(training_set_scaled[i-30:i])
    y.append(y_set[i])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 5))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

hidden_layer_sizes = [150,150]
def build_model(hidden_layer_sizes,act,input_d):
  model = Sequential()

  model.add(LSTM(units = hidden_layer_sizes[0], return_sequences= True,input_shape =input_d))
  model.add(Dropout(0.4))

  model.add(LSTM(units=hidden_layer_sizes[len(hidden_layer_sizes)-1]))
  model.add(Dropout(0.4))

  model.add(Dense(units=25))
  model.add(Dense(units=1,activation="softmax"))

  model.compile(optimizer='adam', loss='mean_squared_error')
  return model




regressor = build_model(hidden_layer_sizes,"relu",(X.shape[1],5))


#callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             #ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

model = regressor.fit(X, y, epochs = 100, batch_size = 100)
plt.plot(model.history['loss'])
plt.show()

###

real_stock_price = data_2['movement'].values

dataset_total = data.iloc[:,0:-1]
inputs = dataset_total[len(dataset_total) - len(data_2) -30:].values
#inputs = inputs.reshape(-1,1)
inputs = sc.fit_transform(inputs)

X_test = []
for i in range(30, len(inputs)):
    X_test.append(inputs[i-30:i])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))
predicted_stock_price= regressor.predict(X_test)
predicted_stock_price = np.reshape(predicted_stock_price, -1)
print('Accuracy', accuracy_score(real_stock_price, predicted_stock_price))
print('Recall:', recall_score(real_stock_price, predicted_stock_price))
print('Precision:', precision_score(real_stock_price, predicted_stock_price))
print("Kappa = {:.2f}".format(cohen_kappa_score(real_stock_price, predicted_stock_price)))
print("Log Loss = {:.2f}".format(log_loss(real_stock_price, predicted_stock_price)))
print('f1:', f1_score(real_stock_price, predicted_stock_price))
print('Confusion Matrix:\n', confusion_matrix(real_stock_price, predicted_stock_price))



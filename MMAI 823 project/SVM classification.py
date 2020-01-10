import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score , recall_score, f1_score, confusion_matrix,cohen_kappa_score,log_loss
import pandas_datareader.data as pd_reader

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
data = data[['Adj Close']]
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
X = np.reshape(X, (X.shape[0], X.shape[1]))

real_stock = data_2['movement'].values

dataset_total = data.iloc[:,0:-1]
inputs = dataset_total[len(dataset_total) - len(data_2) -30:].values
#inputs = inputs.reshape(-1,1)
inputs = sc.fit_transform(inputs)
X_test = []
for i in range(30, len(inputs)):
    X_test.append(inputs[i-30:i])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

svm_clf = SVC(kernel="poly", C=0.025)
svm_clf.fit(X, y)

y_pred_svm = svm_clf.predict(X_test)
print('Accuracy', accuracy_score(real_stock, y_pred_svm))
print('Recall:', recall_score(real_stock, y_pred_svm))
print('Precision:', precision_score(real_stock, y_pred_svm))
print("Kappa = {:.2f}".format(cohen_kappa_score(real_stock, y_pred_svm)))
print("Log Loss = {:.2f}".format(log_loss(real_stock, y_pred_svm)))
print('f1:', f1_score(real_stock, y_pred_svm))
print('Confusion Matrix:\n', confusion_matrix(real_stock, y_pred_svm))
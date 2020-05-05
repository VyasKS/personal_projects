# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Predicting stock price using a type of RNN called LSTM which is proved to be best for time series modeling.
#Let's import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Defining essential functions
def plot_predictions(test, predicted):
    plt.plot(test, color='red',label='Actual Google Price',figsize=(12,8))
    plt.predicted(predicted, color='blue', label='Predicted Google Price')
    plt.title('Google Stock price')
    plt.legend()
    plt.show()
    
def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The RMSE is {}".format(rmse))
    

data = pd.read_csv('/home/sai_vyas/Documents/Kaggle/Data Sets/DJIA_30_Time_Series/stock-time-series-20050101-to-20171231/GOOGL_2006-01-01_to_2018-01-01.csv',
                  index_col='Date',parse_dates=['Date'])
data.head()
data.shape
#Missing values check
train_set = data[:'2016'].iloc[:,1:2].values
test_set = data['2017':].iloc[:,1:2].values

#Let's consider 'High' attribute for prices
data['High'][:'2016'].plot(figsize=(8,6),legend=True)
data['High']['2017':].plot(figsize=(8,6),legend=True)
plt.legend(['Training set (before 2017)', 'Test set (2017 and beyond)'])
plt.title('Google Stock Price')
plt.show()

#Scaling the data set using MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
train_set_scaled = sc.fit_transform(train_set)
train_set_scaled.max()

# Since LSTMs store long term memory state, we create a data structure with 60 timesteps and 1 output
# So for each element of training set, we have 60 previous training set elements 
X_train,y_train = [],[]
for i in range(60,2768):
    X_train.append(train_set_scaled[i-60:i,0])
    y_train.append(train_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)
#Reshaping X_train for efficient modelling
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
#The LSTM architecture
regressor = Sequential()
#First LSTM layer with Dropout regularization after every layer
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
#Second LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
#Third LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
#Fourth LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
#Final output layer

# Now to get the test set ready in a similar way as the training set.
# The following has been done so first 60 entires of test set have 60 previous values which is impossible to get unless we take the whole 
# 'High' attribute data for processing
data_total = pd.concat((data['High'][:'2016'],data['High']['2017':]),axis=0)
inputs = data_total[len(data_total)-len(test_set)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
regressor.add(Dense(units=1))
#Compiling the RNN
regressor.compile(optimizer='rmsprop',loss='mean_absolute_error')
#Fitting to the train set
regressor.fit(X_train, y_train, epochs=20, batch_size=72)

#Preparing X_test and predicting the prices
X_test = []
for i in range(60,311):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualizing results for LSTM
plot_predictions(test_set, predicted_stock_price)

#Evaluating the model
return_rmse(test_set, predicted_stock_price)

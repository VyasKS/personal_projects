""" This file illustrates stock prediction using an advanced machine learning technique called LSTM ( Long Short Term Me
mory networks ) a type of RNN ( Recurrent Neural Network ). We follow through data preprocessing, basic visualizations,
modeling and error metric analysis. Typically, any stock data can be passed in as an argument and it will process assum-
ing there are no missing values. We use Google stock prices tracked under the famous index Dow Jone's Industrial Average
DJIA which tracks top companies as a basket and is a basic indicator for evaluating the United States economy."""

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import math
from sklearn.metrics import mean_squared_error














# Helper functions
class Helper:
    """ Helper functions that contains operations performed during preprocessing"""

    def __init__(self, test, predicted):
        self.test = test
        self.predicted = predicted

    def plot_predictions(self):
        """ Takes in test set and predicted set and returns a plot of both actual and predicted prices"""
        plt.plot(self.test, color = 'red', label = 'Actual stock price')
        plt.plot(self.test, color = 'blue', label = 'Predicted stock price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

    def RMSE(self):
        r = math.sqrt(mean_squared_error(self.test, self.predicted))
        print(" >>> The root mean squared error is {}".format(r))


# Importing the dataset
dataset = pd.read_csv('GOOGL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
print(dataset.head())

# Missing value inspection
print(dataset.isnull().sum())

# No missing values are present. Data is clean and perfect for modeling
# Splitting data for training and testing (considering day's high as point of interest)
training_set = dataset[:'2016'].iloc[:, 1:2].values
test_set = dataset['2017':].iloc[:, 1:2].values

# Visualizing dataframe with 'High' attribute
dataset["High"][:'2016'].plot(figsize=(16, 4), legend=True)
dataset["High"]['2017':].plot(figsize=(16, 4), legend=True)
plt.legend(['Training set (Before 2017)', 'Test set (2017 and beyond)'])
plt.title('Stock price')
plt.show()

# Scaling the datasets
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# We create a data structure with 60 time steps and 1 output for memory as LSTMs store long term memory
# For each datapoint of training set, we have 60 prior data points in training set
X_train, y_train = [], []
for i in range(60,2768):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping X_train for efficient modeling
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Modeling ( LSTM architecture )
class Model(nn.Module):

    def __init__(self, n_hidden=50, n_layers=5, drop_prob=0.2, lr=0.001):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.lr = lr

        # Defining LSTM
        self.lstm = nn.LSTM(input_size=(X_train.shape[1], 1), n_hidden=n_hidden, n_layers=n_layers, dropout=drop_prob)

        # Define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # Define final fully connected output layer
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x, hidden):
        """ Forward pass through the network
        Args:
            x : inputs
            hidden : state of hidden cell (hidden or not)"""
        # Get output from new hidden state from LSTM
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        return out, hidden

    def init_hidden(self, batch_size):
        """ Initializes hidden state """
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden



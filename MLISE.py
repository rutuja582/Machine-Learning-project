# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 15:29:37 2021

@author: atharv
"""

# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest


#Data
trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

#Calculating total opendays
trainData['Open Date'] = pd.to_datetime(trainData['Open Date'], format='%m/%d/%Y')   
trainData['OpenDays']=""

dateLastTrain = pd.DataFrame({'Date':np.repeat(['01/01/2018'],[len(trainData)]) })
dateLastTrain['Date'] = pd.to_datetime(dateLastTrain['Date'], format='%m/%d/%Y') 
dateLastTrain.head()


trainData['OpenDays'] = dateLastTrain['Date'] - trainData['Open Date']
trainData['OpenDays'] = trainData['OpenDays'].astype('timedelta64[D]').astype(int) 


# Importing the dataset
dataset = pd.read_csv('train.csv')
X = trainData['OpenDays'].values
y = dataset.iloc[:, 42].values

X.reshape(1, -1)
y.reshape(1, -1)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train.reshape(1, -1) , y_train.reshape(1, -1))


# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results 
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Revenue VS obfuscated data (Training set)')
plt.xlabel('obfuscated data')
plt.ylabel('Revenue')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test,y_pred,color='pink')
plt.plot(X_train, regressor.predict(X_train), color = 'yellow')
plt.title('Revenue VS obfuscated (Test set)')
plt.xlabel('obfuscated data')
plt.ylabel('Revenue')
plt.show()
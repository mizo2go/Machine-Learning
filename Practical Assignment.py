# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

import pandas as pd

# Plotting library
from matplotlib import pyplot

data= pd.read_csv("C:/Users/mina/Downloads/house_data_complete.csv")
#print(data)

train, validate, test = np.array_split(data.sample(frac=1), [int(.6*len(data)), int(.8*len(data))])
#print('Train',train)
#print('Validate',validate)
#print('test',test)
#print('Data',data)

#Plotting
fig = pyplot.figure()  # open a new figure
size=train['sqft_living']
price=train['price']

#pyplot.plot(size, price, 'ro', ms=2, mec='y')
#pyplot.ylabel('House price')
#pyplot.xlabel('House size in sqft')
#pyplot.show()

#Normalizing the data
m= train.shape[0]
y=train['price']
X=train.drop(columns=['price', 'date']) #drop lel date 3shan mesh byet7eseblo mean wala std fa dimensions el matrix btedrab
X_norm = X.copy()
mu = np.zeros(X.shape[1]) #initialize mu with same num of columns of train
sigma = np.zeros(X.shape[1]) #initialize sigma with same num of columns of train
mu = np.mean(X, axis=0) #axis 0 ya3ny el columns w 1 ya3ny el rows
sigma = np.std(X, axis=0)
X_norm = (X - mu) / sigma

#hypothesis
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)
initial_theta = np.zeros(X.shape[1])
h1 = np.dot(X, initial_theta)
lambda_ = 1

def costFunctionReg(theta, X, y, h, lambda_):
    m= train.values[:,2].size
    J= np.dot((h - y), (h - y)) / (2 * m) + ((lambda_/(2 * m))* np.sum(np.dot(theta, theta)))
    return J

cost = costFunctionReg(initial_theta, X, y, h1, lambda_)

print('Cost at initial theta (zeros): {:.3f}'.format(cost))

def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]  # number of training examples
    theta = theta.copy()
    J_history = []  # Use a python list to save cost in every iterationv

    for i in range(num_iters):
        alphabym = alpha / m
        h = np.dot(X, theta)
        theta = theta*(1 - (alpha*lambda_)/m) - ((alpha / m) * (np.dot(X.T, h - y)))
        J_history.append(costFunctionReg(theta, X, y, h, lambda_))

    return theta, J_history
def gradientDescent2(X, y, theta, alpha, num_iters):
    m = y.shape[0]  # number of training examples
    theta = theta.copy()
    J_history = []  # Use a python list to save cost in every iterationv

    for i in range(num_iters):
        alphabym = alpha / m
        h = np.dot(np.power(X,2), theta)
        theta = theta*(1 - (alpha*lambda_)/m) - ((alpha / m) * (np.dot(X.T, h - y)))
        J_history.append(costFunctionReg(theta, X, y, h, lambda_))

    return theta, J_history

def gradientDescent3(X, y, theta, alpha, num_iters):
    m = y.shape[0]  # number of training examples
    theta = theta.copy()
    J_history = []
    k = X.copy()
    k[:, 4] = np.power(k[:, 4], 2)# Use a python list to save cost in every iterationv

    for i in range(num_iters):
        alphabym = alpha / m
        h = np.dot(k, theta)
        theta = theta*(1 - (alpha*lambda_)/m) - ((alpha / m) * (np.dot(X.T, h - y)))
        J_history.append(costFunctionReg(theta, X, y, h, lambda_))

    return theta, J_history

iterations = 100
alpha = 0.1
alpha2 = 0.003
theta, J_history = gradientDescent(X,y, initial_theta, alpha, iterations)
theta2, J_history2 = gradientDescent2(X,y, initial_theta, alpha2, iterations)
theta3, J_history3 = gradientDescent3(X,y, initial_theta, alpha, iterations)
print(theta)
print('J_history',J_history)
pyplot.plot(np.arange(100), J_history, '-', ms=2, mec='y')
pyplot.plot(np.arange(100), J_history2, '-', ms=2, mec='y')
pyplot.plot(np.arange(100), J_history3, '-', ms=2, mec='y')
pyplot.ylabel('error')
pyplot.xlabel('iterations')
pyplot.show()

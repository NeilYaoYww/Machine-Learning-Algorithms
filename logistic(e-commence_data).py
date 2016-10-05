# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:04:45 2016

@author: WYAO2
"""
"""
Logistic Regression on the E-commence data
"""
import os
os.chdir("C:\\Users\\WYAO2\\Documents\\Udemy\\Logistic Regression")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from process import get_binary_data
from logistic import gradient_descent

# get the data
X, Y = get_binary_data()
X, Y = shuffle(X, Y)

# create train and test data
Xtrain = X[:-100]
Ytrain = X[:-100]
Xtest = X[-100:]
Ytest = Y[-100:]

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

def classification_rate(Y, P):
    return np.mean(Y == P)
    
def cross_entropy(T, pY):        
    return -np.mean(T*np.log(pY) + (1-T)*np.log(1-pY))
    
# creat two lists to store training cost and testing cost
train_cost = []
test_cost = []
            
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:35:13 2016

@author: WYAO2
"""

"""
Cross_Entropy and Regularization
"""

import os
os.chdir("C:\\Users\\WYAO2\\Documents\\Udemy\\Logistic Regression")
import numpy as np
import pandas as pd
from Sigmoid import sigmoid


"""
First, create two normal distributed clusters with different means,
Assign one cluster to class 1 and the rest to class 0
"""
N = 100
D = 2
X = np.random.randn(N,D) # create a data with 2 columns and 100 rows
X[:50,:] = X[:50,:] - 2*np.ones((50,D)) # centered at [-2, -2]
X[50:,:] = X[50:,:] + 2*np.ones((50,D)) # centerted at [2, 2]
T = np.array([0]*50 + [1]*50)
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X),axis=1)

#randomly initialize the weights
w = np.random.randn(D+1)
#calculate the model output
z = Xb.dot(w)

def cross_entropy(T, Y): # calcualte the cross_entropy (error measure)
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1-Y[i])          
    return E
            
            
#calculate the sigmoid
Y = sigmoid(z)
            
"""
Use gradient descent
"""         
def gradient_descent(T,Y,I,L,w): 
# T = true classes, Y = predicted probablity, 
# I = number of iterations, L = learning rate
# w = initial weights
    for i in range(I):
        if i%10 == 0:
            print (cross_entropy(T, Y))
        w += L * np.dot((T-Y).T, Xb)
        Y = sigmoid(Xb.dot(w))
    print ("Final w:", w)
    
gradient_descent(T,Y,1000,0.1,w)
            
            
            

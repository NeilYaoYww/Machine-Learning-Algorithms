# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


"""
Creat the Sigmoid Function: 
    -a mathematical function having an "S" shaped curve
    -return values within [0,1]
    -y intercept 0.5
"""
import numpy as np
N=100
D=2
X = np.random.randn(N,D)
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones,X), axis=1)
w = np.random.randn(D + 1)

z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

print (sigmoid(z))

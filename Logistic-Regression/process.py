# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:17:19 2016

@author: WYAO2
"""

"""
Pre-process the Data
    -split x and y
    -normalize numeric columns and create dummys for categorical variables
    -only take two classes in y as we are doing a binary classification
"""

import pandas as pd
import numpy as np

def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.as_matrix()
    X = data[:,:-1]
    Y = data[:,-1]
    
    #normalize numeric columns
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()
    
    # create a new matrix X2 with the correct number of columns
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)] # non-categorical

    # one-hot
    for n in range(N):
        t = int(X[n,D-1])
        X2[n,t+D-1] = 1

    # method 2
    # Z = np.zeros((N, 4))
    # Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    # # assign: X2[:,-4:] = Z
    # assert(np.abs(X2[:,-4:] - Z).sum() < 10e-10)

    return X2, Y


def get_binary_data():
    # return only the data from the first 2 classes
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2
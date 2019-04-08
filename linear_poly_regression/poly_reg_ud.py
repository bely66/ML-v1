# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:44:15 2019

@author: mohamed nabil 
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

data = pd.read_csv("data_ud.csv")

X = data.iloc[:,:-1]
Y = data.iloc[:,-1]



# Create polynomial features
from sklearn.preprocessing import PolynomialFeatures

# predictor feature

poly_feat = PolynomialFeatures(4)
x_poly = poly_feat.fit_transform(X)

# Make and fit the polynomial regression model
from sklearn.linear_model import LinearRegression 
reg = LinearRegression(fit_intercept=False).fit(x_poly,Y)



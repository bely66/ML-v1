# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:13:40 2019

@author: mohamed nabil
"""
# TODO: Add import statements
import numpy as np 
import pandas as pd 


# Assign the data to predictor and outcome variables

train_data = pd.read_csv("data_reg.csv")
X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(X)
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)

# : Fit the model.
lasso_reg.fit(x,y)
# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)

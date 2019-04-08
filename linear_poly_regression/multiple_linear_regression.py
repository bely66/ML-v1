# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:37:34 2019

@author: mohamed_nabil
"""
# always check for assumptions of linear regression
# 1- linearity / 2- homoscedasticity
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#loading the dataset
file = pd.read_csv("50_Startups.csv")

#definig the independent/dependent variables
X = file.iloc[:,:-1].values
Y = file.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
le_x = LabelEncoder()
le_x=le_x.fit(X[:,3])
X[:,3]=le_x.transform(X[:,3])
encode = OneHotEncoder(categorical_features=[3]) #enter the index number
X = encode.fit_transform(X).toarray()

#avoiding the dummy variable trap 
X= X[:,1:]
#split the dataset to training and testing set
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.2,random_state=0)
#
from sklearn.linear_model import LinearRegression

regress = LinearRegression()
regress.fit(X_train,Y_train)

#predicting the test set result 
y_pred = regress.predict(X_test)

#using backward elimination to optimize the model 
import statsmodels.formula.api as sm
#append using numpy.append / numpy.ones / astype to change the variables type
X = np.append(arr = np.ones((50,1)).astype(int),values=X, axis=1)

# the optimal model matrix 
X_opt = X[:,[0,1,2,3,4,5]]
regress_ols = sm.OLS(endog=Y,exog=X_opt).fit()
regress_ols.summary()
#remove index 1,2 ,4
X_opt = X[:,[0,3,5]]
regress_ols = sm.OLS(endog=Y,exog=X_opt).fit()
regress_ols.summary()



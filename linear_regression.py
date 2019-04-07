# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:57:41 2019

@author: USER
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#loading the dataset
file = pd.read_csv("Salary_Data.csv")
#definig the independent/dependent variables
X = file.iloc[:,:-1].values
Y = file.iloc[:,1].values

#split the dataset to training and testing set
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 1/3,random_state=0)
# fitting the training set into the model  
 
from sklearn.linear_model import LinearRegression

regress = LinearRegression()
regress = regress.fit(X_train,Y_train)

#predict the test set 
y_pred = regress.predict(X_test)

# visualising the results
plt.scatter(X_train,Y_train,color="red")
plt.scatter(X_test,Y_test,color="blue")

plt.plot(X_train,regress.predict(X_train),color="yellow")
plt.title("salary vs exp")
plt.xlabel("years of exp")
plt.ylabel("salary")
plt.show()

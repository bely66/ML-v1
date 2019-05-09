# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:25:05 2019

@author: mohamed nabil
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('bmi_data.csv')

X = data.iloc[:,1:-1]
Y= data.iloc[:,-1]

from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(X,Y)

y_pred = regress.predict(X)

plt.scatter(X,Y,color='red')
plt.plot(X,y_pred,color='yellow')
plt.show()


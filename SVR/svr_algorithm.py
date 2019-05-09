# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 02:50:11 2019

@author: mohamed nabil
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
file = pd.read_csv("Position_Salaries.csv")

x = file.iloc[:,1:2].values
y= file.iloc[:,2:].values

#applying feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x = sc_X.fit_transform(x)
sc_Y = StandardScaler()
y = sc_Y.fit_transform(y)
from sklearn.svm import SVR
# you should use a suitable kernel
model = SVR(kernel='rbf')
model.fit(x,y)


y_pred = model.predict(x)

plt.scatter(sc_X.inverse_transform(x),sc_Y.inverse_transform(y),color="blue")
plt.plot(sc_X.inverse_transform(x),sc_Y.inverse_transform(model.predict(x)),color="yellow")
plt.show()
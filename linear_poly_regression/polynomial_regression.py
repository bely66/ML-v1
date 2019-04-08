# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 22:37:15 2019

@author: mohamed nabil 
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
data = pd.read_csv("Position_Salaries.csv")
X = data.iloc[:,1:-1].values
Y = data.iloc[:,2].values
# fit linear regression model
from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(X,Y)
y_p1 = regress.predict(X)

#fit polynomial regression model 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
reg_poly = LinearRegression()
reg_poly.fit(X_poly,Y)
y_p2 = reg_poly.predict(poly_reg.fit_transform(X))
#visualizing the linear model 
plt.scatter(X,Y,color="blue")

plt.plot(X,y_p1,color="yellow")

plt.show()

#visualising the polynomial model 
plt.scatter(X,Y,color="red")

plt.plot(X,reg_poly.predict(poly_reg.fit_transform(X)),color="blue")

plt.show()



# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 03:42:15 2019

@author: mohamed nabil
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

file = pd.read_csv("Position_Salaries.csv")

x = file.iloc[:,1:2].values
y= file.iloc[:,2:].values

#fitting the model 
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x,y)


#ploting the decision tree prediction in high resolution
X_grid = np.arange(min(x),max(x),0.01)
X_grid= X_grid.reshape((len(X_grid),1))

plt.scatter(x,y,color="blue")
plt.plot(X_grid,model.predict(X_grid),color="yellow")
plt.show()
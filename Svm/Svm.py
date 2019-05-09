# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 22:55:26 2019

@author: USER
"""

import torch

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

data = pd.read_csv("data.csv")
x = data.iloc[:,:-1]
y=data.iloc [:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(x,y,test_size=0.2)

from sklearn.svm import SVC

model = SVC(kernel='rbf', gamma=19)
model.fit(X_train,y_train)



prediction = model.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction))

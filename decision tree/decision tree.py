# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 01:27:33 2019

@author: mohamed nabil
"""

# Import statements 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]


model = DecisionTreeClassifier(max_depth=7,min_samples_leaf =2)
model.fit(X,y)


y_pred = model.predict(X)


acc = accuracy_score(y,y_pred)
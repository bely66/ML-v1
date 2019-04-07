# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:52:10 2019

@author: mohamed_nabil
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#loading the dataset
file = pd.read_csv("Data.csv")
#definig the independent/dependent variables
X = file.iloc[:,:-1].values
Y = file.iloc[:,3].values

#dealing with missing data 
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values = 'NaN',strategy='mean',axis=0) # strategy = "mean"/"median"/"most_frequent"
#fit the data 
##imp =imp.fit(X[:,1:3])
#make the changes to the data 
##X[:,1:3] = imp.transform(X[:,1:3])

# dealing with categorical variables 
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
le_x = LabelEncoder()
le_x=le_x.fit(X[:,0])
X[:,0]=le_x.transform(X[:,0])
encode = OneHotEncoder(categorical_features=[0])
X = encode.fit_transform(X).toarray()
le_Y = LabelEncoder()
le_Y=le_Y.fit_transform(Y)

#split the dataset to training and testing set
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.2,random_state=0)

#feature scaling 
from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

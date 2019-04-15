# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 01:59:04 2019

@author: mohamed nabil
"""

import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames



# Set a random seed
import random
random.seed(42)

# Load the dataset
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

# Print the first few entries of the RMS Titanic data
display(full_data.head())
inputs = full_data.drop('Survived',axis=1)
outcome = full_data['Survived']
features = pd.get_dummies(inputs)
features = features.fillna(0.0)
display(features.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, outcome, test_size=0.2, random_state=42)
# Import the classifier from sklearn
from sklearn.tree import DecisionTreeClassifier

#improved the model accuracy using the right hyper parameters
model = DecisionTreeClassifier(max_depth=7,min_samples_split=15,min_samples_leaf =6)
model.fit(X_train,y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate the accuracy
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)
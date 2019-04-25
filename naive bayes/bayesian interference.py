# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:21:30 2019

@author: USER
"""

import numpy as np
import pandas as pd

df = pd.read_table( 'smsspamcollection/SMSSpamCollection',sep='\t',names=['label','sms_message'])#TODO



df.head()
df['label'] = df.label.map({'ham':0,'spam':1})

documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']


lower_case_documents = []
for i in documents:
   #convert to lower case
    lower_case_documents.append(i.lower())
print(lower_case_documents)

sans_punctuation_documents = []
import string

for i in lower_case_documents:
    # remove punctuations 
   sans_punctuation_documents.append(i.translate(str.maketrans('', '', string.punctuation)))
print(sans_punctuation_documents)

preprocessed_documents = []
for i in sans_punctuation_documents:
    
    preprocessed_documents.append(i.split())
print(preprocessed_documents)

frequency_list = []
import pprint
from collections import Counter

for i in preprocessed_documents:
    frequency_list.append(Counter(i))
    
pprint.pprint(frequency_list)

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
d=count_vector.fit_transform(documents)
doc_array = d.toarray()

frequency_matrix = pd.DataFrame(doc_array,columns=count_vector.get_feature_names())

#implementing our dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))
data_vector = CountVectorizer()
trainig_set = data_vector.fit_transform(X_train).toarray()
testing_data = data_vector.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(trainig_set,y_train)
prediction = model.predict(testing_data)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print ("accuracy = {}",accuracy_score(y_test,prediction))
print ("percision = {}",precision_score(y_test,prediction))
print ("sensitivity = {}",recall_score(y_test,prediction))
print ("f1 = {}",f1_score(y_test,prediction))
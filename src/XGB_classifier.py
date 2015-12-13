import xgboost as xgb

import os
import re

import theano

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

train = pd.read_json('../data/train.json/train.json')
train.head()

vectorizer = CountVectorizer(max_features = 2000)
ingredients = train['ingredients']
words_list = [' '.join(x) for x in ingredients]

#Make label encoder
le = preprocessing.LabelEncoder()
le.fit(train["cuisine"])

#create a bag of words and convert to a array and then print the shape
bag_of_words = vectorizer.fit(words_list)
bag_of_words = vectorizer.transform(words_list).toarray()
print(bag_of_words.shape)

vectorizertfidf = TfidfVectorizer(min_df=1)
tfidf = vectorizertfidf.fit_transform(words_list).toarray()
print tfidf.shape

X = bag_of_words
y = le.transform(train["cuisine"])

print X.shape
print y.shape

dtrain = xgb.DMatrix(X, label=y)

n_classes = len(list(set(y)))

param = {'max_depth':14, 
         'eta':1, 
         'objective':'multi:softmax',
         'num_class':n_classes }

num_round = 64
bst = xgb.train(param, dtrain, num_round)

p = bst.predict(dtrain).astype(int)

incorrect = 0
for i in xrange(p.shape[0]):
    if(p[i]!= y[i]):
        incorrect += 1

print float(incorrect) / float(p.shape[0])

#Now read the test json file in 
test = pd.read_json('../data/test.json/test.json')
test.head()

#Do the same thing we did with the training set and create a array using the count vectorizer. 
test_ingredients = test['ingredients']
test_ingredients_words = [' '.join(x) for x in test_ingredients]
test_ingredients_array = vectorizer.transform(test_ingredients_words).toarray()

dtest = xgb.DMatrix(test_ingredients_array)
result = bst.predict(dtest)

result = le.inverse_transform(result.astype(int))
# Copy the results to a pandas dataframe with an "id" column and
# a "cusine" column
output = pd.DataFrame( data={"id":test["id"], "cuisine":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "XGB_model.csv", index=False, quoting=3 )
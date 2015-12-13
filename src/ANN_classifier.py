#Import Packages we will be using
import os
import re

import theano

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

from sklearn import preprocessing

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

np.random.seed(1337)  # for reproducibility

train = pd.read_json('../data/train.json/train.json')
train.head()

#Initalize a CountVectorizer only considering the top 2000 features. 
#Then Extract the ingredients and convert them to a single list of recipes called words_list
vectorizer = CountVectorizer(max_features = 2000)
ingredients = train['ingredients']
words_list = [' '.join(x) for x in ingredients]

#create a bag of words and convert to a array and then print the shape
bag_of_words = vectorizer.fit(words_list)
bag_of_words = vectorizer.transform(words_list).toarray()
print(bag_of_words.shape)

batch_size = 128
nb_epoch = 40

X_train = bag_of_words
y_train = le.transform(train["cuisine"])

nb_classes = len(le.classes_)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)

#output_dim=64, input_dim=100,
model = Sequential()
model.add(Dense(1024,input_shape=(2000,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(.4))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

model.fit(X_train, Y_train,
          batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, validation_split=.2, verbose=2)

#Do the same thing we did with the training set and create a array using the count vectorizer. 
test_ingredients = test['ingredients']
test_ingredients_words = [' '.join(x) for x in test_ingredients]
test_ingredients_array = vectorizer.transform(test_ingredients_words).toarray()

result = model.predict_classes(test_ingredients_array)
result = le.inverse_transform(result)

# Copy the results to a pandas dataframe with an "id" column and
# a "cusine" column
output = pd.DataFrame( data={"id":test["id"], "cuisine":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "ANN_model.csv", index=False, quoting=3 )
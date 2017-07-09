# Part 1 - Data Preprocessing

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Import the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data

# set geography col to numerical values
label_encoder_X_1 = LabelEncoder()
X[:, 1] = label_encoder_X_1.fit_transform(X[:, 1])

# set gender col to numerical values
label_encoder_X_2 = LabelEncoder()
X[:, 2] = label_encoder_X_2.fit_transform(X[:, 2])

# ensure not ordinal, just categorical. creates 3 columns representing categories
one_hot_encoder = OneHotEncoder(categorical_features = [1])
X = one_hot_encoder.fit_transform(X).toarray()

# now, remove one of those multilinear category columns to avoid the dummy variable trap
X = X[:, 1:]

# split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Make an ANN

import keras
from keras.models import Sequential
from keras.layers import Dense

# initializing the ANN
classifier = Sequential()

# add the input layer and the first hidden layer
# uses recommended output dimension units which is the average of the number of inputs and outputs, in this
# case (11 + 1)/2 = 6. uses the rectifier activation function.
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# add the second hidden layer
# input_dim only required on the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# add the output layer
# output for this problem is only 1. change the activation to the sigmoid function which returns the probability of outcome
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# copmile the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fit the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)


# Part 3 - Making predictions and evaluating the model
from sklearn.metrics import confusion_matrix

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# in building out our prediction assignment, we cannot use the same data
# manipulations as above for X because the end result would not match. for
# example, there are not 3 country categories in this single entry, and it can't
# how to categorize that.
my_person = sc.fit_transform(np.array([[0, 0, 600, 0, 40, 3, 60000, 2, 1, 1, 50000]]))
new_pred = classifier.predict(my_person)
new_pred = (new_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)

# total correct predictions divided by total rows in the test set
accuracy = (cm[0][0] + cm[1][1]) / X_test.shape[0]


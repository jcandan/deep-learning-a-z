# Training an ANN with Stocastic Gradient Decent
#
# Step 1: Randomly initialize the weights to small numbers close to 0, but not 0
# Step 2: Input the first observation of the dataset in your input layer, each feature in one input node
# Step 3: Forward Propagation: from left to right, the neurons are activated in a way that the impact of each neuron's
#         activation is limited by the weights. Propegate the activations until getting the predicted result y.
# Step 4: Compare the predicted result to the actual result. Measure the generated error.
# Step 5: Back Propagation: from right to left, the error is back-propagated. Update the weights according to how much
#         they are responsible for the error. The learning rate decides by how much we update the weights.
# Step 6: Reinforcement Learning or Batch Learning
#         Repeat steps 1 to 5 and update the weights after each observation or after a batch of operations.
# Step 7: When the entire training set has passed through the ANN, that makes an epoch. Do more epochs.

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


# grab our data
dataset = pd.read_csv("Churn_Modelling.csv")

# note: 3:13 is from 3 to 13, 13 not included
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

###
# 1. Data Preprocessing

# set geography and gender cols to numerical values
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
X[:, 2] = le.fit_transform(X[:, 2])

# In this dataset, Geography has 3 categories (France, Spain, Germany as 0,1,2). We must ensure not considered ordinal
# onehotencoder will split those into 3 columns of (0,1)
ohe = OneHotEncoder(categorical_features = [1])
X = ohe.fit_transform(X).toarray()

# remove one of those category columns to avoid the
# dummy variable trap
X = X[:, 1:]

# setup our train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# perform feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

###
# 2. Build the ANN

# initialize the ANN
classifier = Sequential()

# add the input layer and first hidden layer
# units will be the calculated average of inputs and outputs (11 + 1) / 2 = 6
# activation will be set to rectifier activation function
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# add the second hidden layer
# input_dim only needed on first hidden layer
# keep other properties the same
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# add the output layer
# there's only 1 output for this problem.
# we'll use the sigmoid function for the activation function so it returns a probability
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# compile our ANN - apply stocastic gradient decent
# optimizer set to adam, an algorithm that will try to find the best weights
# loss corresponds to the loss function that is used to find the optimal weights (sum of the squared error)
# metrics is a criterion to evaluate our model, used to improve on each iteration
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fit the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

###
# 3. Evaluating the model

# predict against the test data
y_pred = classifier.predict(X_test)  # our list of predicted probabilities of a customer's exit

# we will build a confusion-matrix on test outcomes vs. the predicted outcomes, however, this requires that the
# predicted outcomes are boolean values, not probabilities.
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)

# from a confusion matrix, one can capture the resulting accuracy from the following:
# total correct predictions divided by the total rows in the test set
accuracy = (cm[0][0] + cm[1][1]) / X_test.shape[0]

###
# 4. Making predictions

# in building out our prediction assignment, we cannot use the same data encoder as above for X because the end result
# would not match. for example, there are not 3 country categories in this single entry, and it can't know how to
# categorize that.
# we do however need to normalize the data using our StandardScaler() defined above
my_person = sc.fit_transform(np.array([[0, 0, 600, 0, 40, 3, 60000, 2, 1, 1, 50000]]))
new_pred = classifier.predict(my_person)
new_pred = (new_pred > 0.5)

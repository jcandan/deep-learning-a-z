import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def build_classifier(units, optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

if __name__ == "__main__":
    # grab our data
    dataset = pd.read_csv("Churn_Modelling.csv")

    # note: 3:13 is from 3 to 13, 13 not included
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values

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

    # get the initial timestamp before attempting the ANN
    time_0 = datetime.datetime.now()

    classifier = KerasClassifier(build_fn = build_classifier)
    parameters = {'batch_size': [32],
                  'epochs': [500],
                  'optimizer': ['rmsprop'],
                  'units': [6, 8, 10]}
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10,
                               n_jobs = -1)
    grid_search = grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_

    # grab the difference in time it took to run
    time_taken = datetime.datetime.now() - time_0

    print('###########################################################')
    print('###########################################################')
    print('###########################################################')
    print('')
    print('')
    print('')
    print('Best Accuracy: ' + str(best_accuracy))
    print('Best Params: ' + str(best_parameters))
    print('Time: ' + str(time_taken))
    print('')
    print('')
    print('')
    print('###########################################################')
    print('###########################################################')
    print('###########################################################')

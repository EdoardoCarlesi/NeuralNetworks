import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

"""
    ANN Classifier builder
"""
def build_classifier():
    classifier = Sequential()

    # First hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
    
    # ----> DROPOUT: random deactivate some neurons to prevent overfitting. 
    # Use higher values of p to prevent overfitting, but not higher than 0.5, otherwise we deactivate most of the neurons and run into underfitting.
    classifier.add(Dropout(p = 0.1))

    # Second hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))

    # Output layer (choose 'softmax' for more than one class)
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier


def tune_classifier(optimizer):
    classifier = Sequential()

    # First hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
    
    # ----> DROPOUT: random deactivate some neurons to prevent overfitting. 
    # Use higher values of p to prevent overfitting, but not higher than 0.5, otherwise we deactivate most of the neurons and run into underfitting.
    classifier.add(Dropout(p = 0.1))

    # Second hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))

    # Output layer (choose 'softmax' for more than one class)
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier



data_file = 'data_ann/Churn_Modelling.csv'
dataset = pd.read_csv(data_file)
print(dataset.info())

# Data preprocessing
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


"""
    K Fold cross validation:
    Take a dataset, divide into cv = 10, use 9 of these as training + 1 as validation; 
    Repeat the operation and use 1 of these subset at the time
    I will end up with a more robust measure of the accuracies, on 10 samples
"""
'''
# We need to build a classifier with Keras to be able to do cross validation
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)

# K-fold cross validation   ---> n_jobs = -1 uses all the availabe CPUs
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

print(accuracies)
print(accuracies.mean(), accuracies.std())
'''

"""
    Grid Search Cross Validation:
    We will tune the hyperparameters on a grid of different batch sizes, epochs etc.
"""

# The batch_size and nb_epoch will be chosen on the grid
classifier = KerasClassifier(build_fn = tune_classifier)

# parameter is a dictionary of all possible hyperparameters
parameters = {'batch_size':[5, 30], 'nb_epoch':[10, 50], 'optimizer':['adam', 'rmsprop']}

# Create the grid search object
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 5)

# Fit on the grid using the training set
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_parameters)
print(best_accuracy)




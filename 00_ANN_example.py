import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

import keras
from keras.layers import Dense
from keras.models import Sequential


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(X[1, :])

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set, create your classifier here
classifier = Sequential()

# ---> Rule of Thumb: N of nodes in a hidden layer is the average between the input layer and the output layer <--- #

# Input layer
# OLD SYNTAX
#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
#classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'relu', input_dim = 6))

# First hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))

# Second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Output layer (choose 'softmax' for more than one class)
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compile the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 5)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Convert to binary
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

vals = [600, 0, 1, 40, 3, 60000.0, 2, 1, 1, 50000]

x_new = np.array([vals])

y_new = classifier.predict(x_new)
print(y_new)






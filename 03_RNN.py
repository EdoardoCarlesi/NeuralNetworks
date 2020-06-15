'''
    RECURRENT NEURAL NETWORKS
'''

import keras as ks
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Data files
data_path = 'data/RNN/'
file_train = 'Google_Stock_Price_Train.csv'
file_test = 'Google_Stock_Price_Test.csv'

'''
    PART 1: Data pre-processing
'''
path_data_train = data_path + file_train
path_data_test = data_path + file_test

data_train = pd.read_csv(path_data_train)
data_test = pd.read_csv(path_data_train)
#print(data_train.head())

# Create a NumPy array from the DataFrame
train_set = data_train.iloc[:, 1:2].values
test_set = data_test.iloc[:, 1:2].values
#print(train_set)

# Feature Scaling: standardization (subtract mean value, divide by std dev) normalisation (subtract min, divide by max - min)
# For Sigmoid function normalization works better than standardization
sc = MinMaxScaler(feature_range=(0, 1))
train_set_scaled = sc.fit_transform(train_set)
#print(train_set_scaled)

# Initialize as empty lists
X_train = []
y_train = []

'''
    Number of time steps: is the number of time steps remembered by the RNN. Beware of overfitting (few steps).
    The output a t+1 depends on the N_steps before that. 
    Here 60 steps are uses
'''

n_steps = 60
n_last = len(train_set_scaled)
n_used = n_last - n_steps

print('Time steps looping from ', n_steps, ' to ', n_last)

# Initialize the training set
for i in range(n_steps, n_last):
    X_train.append(train_set_scaled[i-n_steps:i, 0])
    y_train.append(train_set_scaled[i, 0])

# Convert the lists to data structures
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the array to a new dimension (). Here we can add other features (e.g. other stock prices, volume sales, etc.) editing the shape of the input training set
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

'''
    PART 2: Build the RNN (LSTM)
'''



'''
    PART 3: Make predictions and visualize results
'''



plt.show()





















'''
    SELF ORGANIZING MAPS
'''

import tkinter
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

# Data files
data_path = 'data/SOM/'
data_file = 'Credit_Card_Applications.csv'
file_name = data_path + data_file

# Read datafile
dataset = pd.read_csv(file_name)
#print(dataset.head())

# Generate the datasets
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature scaling
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)
#print(X)

# Generate and train the actual Self Organizing Map
n_feat = len(dataset.columns) - 1
n_x = 10; n_y = 10 

print('Generating a ', n_x, 'x', n_y, ' SOM with ', n_feat, ' features.')

som = MiniSom(x = n_x, y  = n_y, input_len = n_feat, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualize the results and plot the SOM



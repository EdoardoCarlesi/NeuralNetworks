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

from pylab import bone, pcolor, colorbar, plot, show

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

'''
    Visualize the results and plot the SOM
'''

# Plot the background
bone()

# Distance map method returns the matrix of all the mean interneuron distances. The frauds are connected to large distances (outliers)
pcolor(som.distance_map().T)
colorbar()

# We need to higlight customers who cheated and got approval, and those who cheated and did not get approval
marker = ['o', 's']
colors = ['r', 'g']

# Loop on all the customers
for i, x in enumerate(X):

    # This method returns the nearest winning node to the x-element
    w = som.winner(x)

    # y[i] has the information about the transaction approval
    plot(w[0] + 0.5, w[1] + 0.5, marker[y[i]], markeredgecolor = colors[y[i]], markerfacecolor = 'None', markersize = 10)

# Find the frauds. For each node there is a list of customers
mappings = som.win_map(X)
dist_map = som.distance_map().T

# Distance threshold to identify outliers
thresh = 0.95
indexes = []

# Loop on the matrix to identify outliers
for ix in range(0, n_x):
    for iy in range(0, n_y):
        elem = dist_map[ix, iy]

        if elem > thresh:
            indexes.append([ix, iy])

#print(indexes)

# ERROR: sometimes one of the threshold nodes will have ZERO customers in it, so the mapping will return an empty list!
for i, ind in enumerate(indexes):
    index = (ind[0], ind[1])
    frauds = mappings[index]

    if i == 0:
        all_frauds = frauds
    else:
        all_frauds = np.concatenate(frauds, all_frauds)

# Try to invert
frauds = sc.inverse_transform(all_frauds)

plt.show()








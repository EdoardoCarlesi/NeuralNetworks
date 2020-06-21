'''
    SELF ORGANIZING MAPS: Case Study with ANN
'''

import tkinter
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from keras.layers import Dense
from keras.models import Sequential


'''
    FIRST PART is like in the old SOM project, to identify in an unsupervised manner the possible frauds
'''

# Data files
data_path = 'data/SOM/'
data_file = 'Credit_Card_Applications.csv'
file_name = data_path + data_file

# Read datafile
dataset = pd.read_csv(file_name)

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

som = MiniSom(x = n_x, y  = n_y, input_len = n_feat, sigma = 1.0, learning_rate = 0.5, random_seed = 45)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

'''
    Visualize the results and plot the SOM
'''

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
thresh = 0.90

# Initialize some empty lists to be used later on
indexes = []
all_frauds = []

# Loop on the matrix to identify outliers. There might be a smarter way of doing this with np.where but this is clear and easy to implement.
for ix in range(0, n_x):
    for iy in range(0, n_y):
        elem = dist_map[ix, iy]

        if elem > thresh:
            indexes.append([ix, iy])

print('Found ', len(indexes), ' nodes above mean distance threshold: ', thresh)

# Beware: sometimes one of the threshold nodes will have ZERO customers in it, so the mapping will return an empty list!
for i, ind in enumerate(indexes):
    index = (ind[0], ind[1])
    frauds = mappings[index]

    print('Node ', i, ' has ', len(frauds), ' frauds.')

    # Append each fraud candidate one-by-one to the list
    for fraud in frauds:
        all_frauds.append(fraud)

print('There are ', len(all_frauds), ' fraud candidates in total.')

# Invert and get the original data
frauds = sc.inverse_transform(all_frauds)
#print(frauds)

'''
    Now build the ANN with supervised deep learning.
    The list of cheating customers will be our dependent variable to train the ANN.
'''

# Create the matrix of features (forget about column 1 - the customer ID)
X_customers = dataset.iloc[:, 1:].values

# Create the dependent variable using the all_frauds 
n_elem = X_customers.shape[0]

print('Initializing ', n_elem, ' dependent variables to 0 or 1.')
y_fraud = np.zeros((n_elem))

# Look for the customer ID and set the corresponding y_fraud variable to 1 if it is in the list
for fraud in frauds:
    this_id = int(fraud[0])
    col_id = dataset.columns[0]

    col_index = dataset[dataset[col_id] == this_id].index
    y_fraud[col_index[0]] = 1

# Sanity check
fraud_indx = np.where(y_fraud == 1)
print('Successfully initialized ', len(fraud_indx[0]), ' elements to 1.')

# Scale the values
ssc = StandardScaler()
X_customers = ssc.fit_transform(X_customers)

# Build the ANN
classifier = Sequential()

# Input layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = n_feat))

# Output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compile the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit the ANN
print('Training the ANN... ')
classifier.fit(X_customers, y_fraud, batch_size = 1, epochs = 2)

# Run the predictions
print('Running the model predictions. ')
y_pred = classifier.predict(X_customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)

# Sort customers by predicted probability of cheating (in descending order)
y_pred = y_pred[y_pred[:,1].argsort()][::-1]

print(y_pred)



import pandas as pd
import numpy as np

# Torch libraries
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Import the data
base_path = 'data/ml-1m/'

'''
movies = pd.read_csv(base_path + 'movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#print(movies.head())

users = pd.read_csv(base_path + 'users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#print(users.head())

ratings = pd.read_csv(base_path + 'ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#print(ratings.head())
'''

# Training - test set split
data_path = 'data/ml-100k/'
train_set = pd.read_csv(data_path + 'u1.base', delimiter = '\t')

# PyTorch uses NumPy arrays - convert the DS
train_set = np.array(train_set, dtype='int')

# Test set
test_set = pd.read_csv(data_path + 'u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype='int')

# Make a matrix with n users and m movies
n_user = max(max(train_set[:, 0]), max(test_set[:, 0]))
n_film = max(max(train_set[:, 1]), max(test_set[:, 1]))

print(f'Allocating a {n_user} x {n_film} matrix')

# Create a list of lists for PyTorch
def convert(data):
    new_data = []

    for id_user in range(1, n_user+1):
        id_film = data[:, 1][data[:, 0] == id_user]
        id_rating = data[:, 2][data[:, 0] == id_user]

        # Allocate the full ratings line - zero if the user gave no rating
        ratings = np.zeros(n_film)

        # Movie IDs start from 1 not 0
        ratings[id_film-1] = id_rating
        new_data.append(list(ratings))

    return new_data

# Convert the train/test sets to a list of lists
train_set = convert(train_set)
test_set = convert(test_set)

# Convert to a pytorch tensor. The FloatTensor class expects a list of lists
train_set = torch.FloatTensor(train_set)
test_set = torch.FloatTensor(test_set)

# Convert the ratings into 0 or 1 outputs
train_set[train_set == 0] = -1
test_set[test_set == 0] = -1

# Negative ratings
train_set[train_set == 1] = 0
test_set[test_set == 1] = 0
train_set[train_set == 2] = 0
test_set[test_set == 2] = 0

# Positive ratings
train_set[train_set >= 3] = 1
test_set[test_set >= 3] = 1

# Create the architecture of the neural network
class RBM():
    
    # Init with number of visible nodes/ hitten nodes
    def __init__(self, nv, nh):

        # Weights - random initialization
        self.W = torch.randn(nh, nv)

        # Bias for the probability of the hidden nodes given the visible nodes
        self.a = torch.randn(1, nh)

        # Bias for the probability of the visible nodes given the hidden nodes
        self.b = torch.randn(1, nv)








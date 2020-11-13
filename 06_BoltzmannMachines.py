import pandas as pd
import numpy as np

# Torch libraries
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


"""
    Define functions and classes used
"""

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

    # For each hidden node activate it according to some probability
    def sample_h(self, x):
        """
            x = visible neuron value (with the ratings)
        """
        
        # mm is the product of two torch tensors
        wx = torch.mm(x, self.W.t())

        # with expand_as() we make sure each bias is assigned to the weights of the mini bias
        activation = wx + self.a.expand_as(wx)

        # Probability that the node is activated given a value for the visible node
        p_h_given_v = torch.sigmoid(activation)
        
        # The Bernoulli sampling works as follows 
        # For each p_h = [0, 1] we pick a random number n = [0, 1], if n < p_h --> we activate the neuron otherwise no
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    # Sample the visible nodes
    def sample_v(self, y):
         
        # mm is the product of two torch tensors
        wy = torch.mm(y, self.W)

        # with expand_as() we make sure each bias is assigned to the weights of the mini bias
        activation = wy + self.b.expand_as(wy)

        # Probability that the node is activated given a value for the visible node
        p_v_given_h = torch.sigmoid(activation)
        
        # The Bernoulli sampling works as follows 
        # For each p_v = [0, 1] we pick a random number n = [0, 1], if n < p_v --> we activate the neuron otherwise no
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    # Use contrastive divergence to train the RBM
    def train(self, v0, vk, ph0, phk):
            """
                v0 = is the input vector with the ratings on one user
                vk = visible nodes after k samplings (iterations of contrastive divergence)
                ph0 = vector of probabilities with the input values
                phk = probabilities of hidden nodes after k iterations, using the visible nodes vk
            """
            
            # Update weights and biases
            self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
            self.b += torch.sum((v0 - vk), 0)  # ----> 0 is added to keep the shape of the tensor b
            self.a += torch.sum((ph0 - phk), 0)

"""
                        Main program
"""

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

# Number of elements for the input (visible) nodes
nv = len(train_set[0])

# For the hidden nodes we have to choose the number of features to detect
nh = 100

# number of "observations" for the update of the weights
batch_size = 100

# Create the object RBM
rbm = RBM(nv, nh)

# Now train the RBM
nb_epochs = 5

# Number of contrastive divergence steps
nk = 10

# Loop on training epochs
for epoch in range(1, nb_epochs + 1):
    train_loss = 0 
    counter = 0.0

    # Loop on the batch of users
    for id_user in range(0, n_user - batch_size, batch_size):
        vk = train_set[id_user:id_user+batch_size]
        v0 = train_set[id_user:id_user+batch_size]
        ph0, _ = rbm.sample_h(v0)
    
        # Contrastive divergence
        for k in range(nk):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)

            # Freeze the nodes where there is no rating i.e. node = -1
            vk[v0 < 0] = v0[v0 < 0]
        
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0 > 0] - vk[v0 > 0]))
        counter += 1.0

    print(f'Epoch: {epoch}, Loss: {train_loss/counter}')

"""
                Now test the trained model
"""

test_loss = 0
counter = 0.0

print(f'Users: {n_user}, starting to test...')

for id_user in range(0, n_user):
    v = train_set
    vt = test_set

    if len(vt[vt >=  0]) > 0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)

    test_loss += torch.mean(torch.abs(vt[vt > 0] - v[vt > 0]))
    counter += 1.0

print(f'Test Loss: {test_loss/counter}')



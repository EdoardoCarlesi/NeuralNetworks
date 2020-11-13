import pandas as pd
import numpy as np

# Torch libraries
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

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

# Create a class for the stacked autoencoder (several hidden layers)
class SAE(nn.Module):

    # Init with some default values
    def __init__(self, n_movies, n_nodes1=20, n_nodes2=10):
    
        # This is taking the init from the parent class
        super(SAE, self).__init__()
        
        # Full connection 
        self.fc1 = nn.Linear(n_movies, n_nodes1)
        self.fc2 = nn.Linear(n_nodes1, n_nodes2)

        # The third and fourth layer start the decoding
        self.fc3 = nn.Linear(n_nodes2, n_nodes1)
        self.fc4 = nn.Linear(n_nodes1, n_movies)

        # Define the activation function
        self.activation = nn.Sigmoid()

    def forward(self, x):

        # First encoded vector
        x = self.activation(self.fc1(x))

        # Second encoded vector
        x = self.activation(self.fc2(x))

        # Decooding
        x = self.activation(self.fc3(x))

        # On the last layer we don't need to add the activation function
        x = self.fc4(x)

        return x


"""
        MAIN PROGRAM
"""


'''
# Import the dataset
base_path = 'data/ml-1m/'

movies = pd.read_csv(base_path + 'movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

users = pd.read_csv(base_path + 'users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

ratings = pd.read_csv(base_path + 'ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
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

# Convert the train/test sets to a list of lists
train_set = convert(train_set)
test_set = convert(test_set)

# Create a structure that contains all the observations for the network
train_set = torch.FloatTensor(train_set)
test_set = torch.FloatTensor(test_set)

print('Creating a Stacked Auto Encoder SAE()...')
sae = SAE(n_film)

# Define a loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Train the SAE
n_epoch = 5

# Loop on the epochs
for epoch in range(1, n_epoch+1):
    
    train_loss = 0.0

    # Counter for the number of steps actually realized
    s = 0.0
    
    # loop on the users
    for id_user in range(n_user):

        # this is creating a batch of a single input vector (add a dimension for pytorch compatibility)
        input = Variable(train_set[id_user]).unsqueeze(0)
        target = input.clone()

        # check if the user actually rate anything
        if torch.sum(target.data) > 0:

            output = sae(input)

            # This ensure it is not computing SGD with respect to the target values
            target.require_grad = False

            # Set to zero the values where the target is zero
            output[target == 0] = 0

            loss = criterion(output, target)

            # Check for the movies that have non zero ratings
            mean_corrector = n_film / float(torch.sum(target.data > 0) + 1.e-10)
            
            # Backpropagation
            loss.backward()

            train_loss += np.sqrt(loss.data * mean_corrector)

            s += 1.0
    
            optimizer.step()
            
    print(f'Epoch {epoch}, loss: {train_loss/s}')

# Move to the test set for the application of the encoder
test_loss = 0.0

# Counter for the number of steps actually realized
s = 0.0
    
# Loop on the users
for id_user in range(n_user):

    # This is creating a batch of a single input vector (add a dimension for pytorch compatibility)
    input = Variable(train_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)

    # Check if the user actually rate anything
    if torch.sum(target.data) > 0:

        output = sae(input)

        # This ensure it is not computing SGD with respect to the target values
        target.require_grad = False

        # Set to zero the values where the target is zero
        output[target == 0] = 0

        loss = criterion(output, target)

        # Check for the movies that have non zero ratings
        mean_corrector = n_film / float(torch.sum(target.data > 0) + 1.e-10)
    
        test_loss += np.sqrt(loss.data * mean_corrector)

        s += 1.0
            
print(f'Test loss: {test_loss/s}')







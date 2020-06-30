import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel 
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Path and filenames
path_dir = 'data/lm-25m/'
path_movies = 'movies.dat'
path_users = 'movies.dat'
path_ratings = 'ratings.dat'

# Import the dataset
movies = pd.read_csv(path_dir + path_movies, sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv(path_dir + path_users, sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv(path_dir + path_ratings, sep = '::', header = None, engine = 'python', encoding = 'latin-1')



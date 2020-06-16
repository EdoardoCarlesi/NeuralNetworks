'''
    RECURRENT NEURAL NETWORKS
'''
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Keras RNN modules
import keras as ks 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Data files
data_path = 'data/RNN/'
file_train = 'Google_Stock_Price_Train.csv'
file_test = 'Google_Stock_Price_Test.csv'

# Train and test datasets
path_data_train = data_path + file_train
path_data_test = data_path + file_test

# Train model or load it 
#trainModel = True
trainModel = False

'''
    PART 1: Data pre-processing
'''

data_train = pd.read_csv(path_data_train)
#print(data_train.head())

# Create a NumPy array from the DataFrame
train_set = data_train.iloc[:, 1:2].values
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

'''
    Reshape the array to a new dimension (). 
    Here we can add other features (e.g. other stock prices, volume sales, etc.) editing the shape of the input training set
    1 here means there is 1 predictor (indicator)
'''

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print('X_train shape: ', X_train.shape)

'''
    PART 2: Build the RNN (LSTM)
'''

# Initialize the RNN
regressor = Sequential()

# Add the first LSTM layer + Dropout regularization. Return sequences = True ---> there is another LSTM layer!
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2)) 

# Second LSTM layer. The input shape is recognized automatically from the previous layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) 

# Third LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) 

# Fourth LSTM layer
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2)) 

# Output layer
regressor.add(Dense(units = 1))

# Compile the RNN, choose optimizer and loss function
regressor.compile(optimizer = "adam", loss = 'mean_squared_error')

# We save the regressor to this output file
n_epochs = 100 #20 #50
file_name = 'regressor_rnn_' + str(n_epochs) + '.keras'

# Fit the RNN to the training set 
if trainModel == True:
    print('Training the RNN...')
    regressor.fit(X_train, y_train, epochs = n_epochs, batch_size = 32)

    regressor.save(file_name)

# We load the RNN model from a pre-saved file
else:
    print('Loading a pre-computed RNN model: ', file_name)
    regressor = ks.models.load_model(file_name)

print('Done.')

'''
    PART 3: Make predictions and visualize results
'''

print('Rescaling test input data...')

# Get the real value (test set): test_set_scaled 
data_test = pd.read_csv(path_data_test)
test_set = data_test.iloc[:, 1:2].values

# We need to concatenate all the datasets --> the test dataset depends on the 60 last train dataset points, and they need to be normalized in the same way
data_total = pd.concat((data_train['Open'], data_test['Open']), axis = 0)
inputs = data_total[len(data_total) - len(data_test) - n_steps:].values

print('Test input data: ', len(test_set))
print('New input data: ', len(inputs))

# Get the predicted value
inputs = inputs.reshape(-1, 1)

# Rescale the input: we need to get the same kind of rescaling that we used for the training set
inputs = sc.transform(inputs)

# Compare the results
X_test = []
n_test_all = len(inputs)

# Only take data in January
n_test = n_steps + 20   

for i in range(n_steps, n_test):
    X_test.append(inputs[i-n_steps:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print('Predict the model using the RNN regressor...')

predicted_price = regressor.predict(X_test)

# Invert the scaling to get real stock prices from the rescaled ones
predicted_price = sc.inverse_transform(predicted_price)

print(predicted_price)

print('Done.')

# Plot the results
plt.plot(test_set)
plt.plot(predicted_price)

plt.show()






















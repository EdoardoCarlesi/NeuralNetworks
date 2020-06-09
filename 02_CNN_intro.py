import pandas as pd
import numpy as np
from keras.models import Sequential

# First CNN step: convolution
from keras.layers import Convolution2D
# Second CNN step: pooling
from keras.layers import MaxPooling2D
# Third CNN step: flattening
from keras.layers import Flatten
# Add an ANN
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

# Initialize the CNN
classifier = Sequential()

# Add the first convolutional layer: 
#   - 32 ----> number of feature detectors, 3x3 dimension. 32 feature maps
#   - input_shape = colored images are converted into 3d arrays (blue, green, red) + number of pixels. BW images have 1 dimension
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Pooling step: reduce the size of the feature maps, specify the size of the subtable
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Second convolutional layer ---> increase the thickness of the convolutional NN to improve the results
classifier.add(Convolution2D(64, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening: take all the pooled feature map and put it into a vector
classifier.add(Flatten())

# Build the fully connected ANN. First hidden layer
classifier.add(Dense(units=128, activation='relu'))

# Output layer
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Increase the data size
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('cnn_dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('cnn_dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)











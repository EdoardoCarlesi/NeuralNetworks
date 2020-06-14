import tensorflow as tf 

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

import keras
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

print('TF Test GPU Available: ', tf.test.is_gpu_available())

print('TensorFlow Version: ')
print(tf.__version__)

print('Keras Version: ')
print(keras.__version__)

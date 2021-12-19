import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import pickle

def relu(x):
    # https://numpy.org/doc/stable/reference/generated/numpy.where.html
    return np.where(x>0, x, 0)

# https://numpy.org/doc/stable/reference/generated/numpy.reciprocal.html
def sigmoid(x):
     return np.reciprocal(1+np.exp(-1*x))

# https://www.tensorflow.org/tutorials/keras/classification
fashion_mnist = tf.keras.datasets.fashion_mnist

# https://github.com/tensorflow/tensorflow/issues/33285
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# https://www.tensorflow.org/tutorials/keras/classification
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#https://ianlondon.github.io/blog/pickling-basics/
#with open('mnist_data.pickle', 'wb') as f:
#    pickle.dump((train_images, train_labels,test_images, test_labels), f)
with open('mnist_data.pickle', 'rb') as f:
    train_images, train_labels,test_images, test_labels = pickle.load(f)

train_images = train_images / 255.0
test_images = test_images / 255.0

vector = np.random.random([10,])-0.5
print(vector)
print(relu(vector))
print(sigmoid(vector))


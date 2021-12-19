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


# https://machinelearningmastery.com/softmax-activation-function-with-python/
def softmax(x):
    exp_x = np.exp(x)
    return exp_x/exp_x.sum()


# https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
# https://realpython.com/gradient-descent-algorithm-python/
def crossentropy_loss(true_y,pred_y):
    part1 = np.matmul(np.log(1 - pred_y), (1 - true_y))
    part2 = np.matmul(true_y, np.log(pred_y))
    return -1*(part1.sum() + part2.sum())





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

y_true = np.random.random([10,])
y_pred = softmax(np.random.random([10,]))
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.round_.html

y_true = np.round(y_true,0)
print(y_true)
print(y_pred)
print(crossentropy_loss(y_true, y_pred))


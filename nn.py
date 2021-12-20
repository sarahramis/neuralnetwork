import random

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import pickle


def same(x):
    return x


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


class Layer:
    def __init__(self, num_inputs, num_neurons,
                 activation_function=relu):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        # each input has a weight going to each neuron
        self.weights = np.random.random(
            [num_inputs,num_neurons])
        self.bias = np.random.random([num_neurons,1])
        self.activation_function = activation_function

    def activation(self, input_values):
        print(self.weights.shape)
        print(input_values.shape)
        z = np.matmul(np.transpose(self.weights),
                      input_values)
        return self.activation_function(z+self.bias)

    def __str__(self):
        return f"Layer of {self.num_neurons} neurons, each with {self.num_inputs} inputs, giving {self.num_inputs*self.num_neurons} weights."


class NeuralNetwork:
    def __init__(self, layer_sizes, layer_activations):
        layers = []
        # assume first layer is an input layer
        for i, layer_size in enumerate(layer_sizes):
            if i > 0:
                # num inputs is number of neurons in
                # previous layer
                num_inputs = layer_sizes[i-1]
                # num_neurons defined by user
                num_neurons = layer_size
                layers.append(Layer(
                    num_inputs=num_inputs,
                    num_neurons=num_neurons,
                    activation_function=layer_activations[i]))
        self.layers = layers


    def __str__(self):
        final_string = "Neural Network:\n"
        for layer in self.layers:
            final_string += str(layer) + "\n"
        return final_string




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

L = Layer(2,1, activation_function=relu)
L.weights[0] = 1
L.weights[1] = -0.5
L.bias = -5
inputs = np.array([1,2])
print(L.activation(inputs))
print(L)
NN = NeuralNetwork([100, 64, 1], [same, relu, sigmoid])
print(NN)
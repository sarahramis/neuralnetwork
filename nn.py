from keras.utils.np_utils import to_categorical
import numpy as np
import tensorflow as tf
import pickle

REDOWNLOAD = False
np.random.seed(26)

# needs to be run with redownload set to True the first time
# this downloads mnist using tensorflow
# as saves it as a python pickle file
# then next time code is run, can set redownload to false
# and it will just load it from local file
def get_mnist(redownload=False):
    if redownload:
        print("Redownloading...")
        # https://www.tensorflow.org/tutorials/keras/classification
        fashion_mnist = tf.keras.datasets.fashion_mnist
        # The below stops an error due to lack of ssl (it's
        # to do with downloading via https
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
        # download
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        # store on local disk
        # https://ianlondon.github.io/blog/pickling-basics/
        with open('mnist_data.pickle', 'wb') as f:
            pickle.dump((X_train, y_train, X_test, y_test), f)

    else:
       # load the MNIST fashion from local disk (faster for development)
        with open('mnist_data.pickle', 'rb') as f:
            X_train, y_train, X_test, y_test = pickle.load(f)
    # tf doesn't provide the class names
    # https://www.tensorflow.org/tutorials/keras/classification
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return X_train, y_train, X_test, y_test, class_names


# Download MNIST or load from local disk
X_train, y_train, X_test, y_test, class_names = get_mnist(redownload=REDOWNLOAD)
#print(X_train.shape)
# reshape data for flat input layer on neural net
X_train = np.reshape(X_train, (X_train.shape[0], 784))
X_test = np.reshape(X_test, (X_test.shape[0], 784))
#print(X_train.shape)

# normalise data
X_train, X_test = (X_train / 255), (X_test / 255)
# convert output data into 10 neuron output layer categories
# one hot encoding.
y_train, y_test = to_categorical(y_train), to_categorical(y_test)


# Classification Neural network class that allows 1 or 2 hidden layers, adjustable numbers
# of neurons in all layers, softmax on output layer,
# and either all relu or all sigmoid on other layers.
# Inspired by https://mlfromscratch.com/neural-network-tutorial/#/
# and https://www.kaggle.com/accepteddoge/fashion-mnist-with-numpy-neural-networks
# and https://github.com/SkalskiP/ILearnDeepLearning.py/blob/master/01_mysteries_of_neural_networks/03_numpy_neural_net/Numpy%20deep%20neural%20network.ipynb
class NumpyNeuralNet():
    def __init__(self, input_size, hidden_sizes, output_size, use_sigmoid=False):
        # check if one or two hidden layers
        # note - hidden_sizes must be a list even if it is just one hidden layer.
        if len(hidden_sizes) > 1:
            self.two_hidden = True
        else:
            self.two_hidden = False
        # store the actual activation function directly in the object
        if use_sigmoid:
            self.activation_function = self.sigmoid
            self.activation_function_back = self.sigmoid_backprop
        else:
            self.activation_function = self.relu
            self.activation_function_back = self.relu_backprop

        # if two hidden layers initialise 3 weight sets (w1i, w2j, w3k),
        # otherwise 2 (w1i, w2j)
        self.p = dict()
        if self.two_hidden:
            # weight initialisation from https://mlfromscratch.com/neural-network-tutorial/#/
            self.p['w1'] = np.random.randn(hidden_sizes[0], input_size) * np.sqrt(1. / hidden_sizes[0])
            self.p['w2'] = np.random.randn(hidden_sizes[1], hidden_sizes[0]) * np.sqrt(1. / hidden_sizes[1])
            self.p['w3'] = np.random.randn(output_size, hidden_sizes[1]) * np.sqrt(1. / output_size)
        else:
            # weight initialisation from https://mlfromscratch.com/neural-network-tutorial/#/
            self.p['w1'] = np.random.randn(hidden_sizes[0], input_size) * np.sqrt(1. / hidden_sizes[0])
            self.p['w2'] = np.random.randn(output_size, hidden_sizes[0]) * np.sqrt(1. / output_size)

    # https://numpy.org/doc/stable/reference/generated/numpy.where.html
    # https://stackoverflow.com/questions/46411180/implement-relu-derivative-in-python-numpy
    def relu(self, x):
        return np.where(x > 0, x, 0)

    def relu_backprop(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_backprop(self, x):
        return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)

    def softmax(self, x):
        exp_x = np.exp(x - x.max())
        return exp_x / np.sum(exp_x, axis=0)

    def softmax_backprop(self, x):
        exp_x = np.exp(x - x.max())
        return exp_x / np.sum(exp_x, axis=0) * (1 - exp_x / np.sum(exp_x, axis=0))

    def activation(self, input_values):
        self.p['a0'] = input_values
        self.p['z1'] = np.dot(self.p['w1'], self.p['a0'])
        self.p['a1'] = self.activation_function(self.p['z1'])
        if self.two_hidden:
            self.p['z2'] = np.dot(self.p['w2'], self.p['a1'])
            self.p['a2'] = self.activation_function(self.p['z2'])
            self.p['z3'] = np.dot(self.p['w3'], self.p['a2'])
            self.p['a3'] = self.softmax(self.p['z3'])
            return self.p['a3']
        else:
            self.p['z2'] = np.dot(self.p['w2'], self.p['a1'])
            self.p['a2'] = self.softmax(self.p['z2'])
            return self.p['a2']



neural_net = NumpyNeuralNet(input_size=784, hidden_sizes=[128, 64],
                            output_size=10, use_sigmoid=True)

print(neural_net.activation(X_train[0])) 

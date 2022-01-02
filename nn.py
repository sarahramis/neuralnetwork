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
# e.g. [1,0,0...0] for shirt, [0,1,0...] for shoes, etc rather than 0 to 9
y_train, y_test = to_categorical(y_train), to_categorical(y_test)


# Classification Neural network class that allows 1 or 2 hidden layers, adjustable numbers
# of neurons in all layers, softmax on output layer,
# and either all relu or all sigmoid on other layers.
# It is only really suitable for classification, because output
# layer is always softmax.
# Inspired by https://mlfromscratch.com/neural-network-tutorial/#/
# and https://www.kaggle.com/accepteddoge/fashion-mnist-with-numpy-neural-networks
# and https://github.com/SkalskiP/ILearnDeepLearning.py/blob/master/01_mysteries_of_neural_networks/03_numpy_neural_net/Numpy%20deep%20neural%20network.ipynb
class NumpyNeuralNet():
    # input_size is number of input neurons, hidden sizes is a list of one
    # For MNIST it's always 784 (flattened picture matrix)
    # or two numbers, giving the number of neurons in up to two
    # hidden layers.
    # output_size the number of classification units.
    # use_sigmoid False means all layters except output layer are Relu
    # otherwise they're sigmoid. Relu is generally preferred and
    # computationally less expensive.
    def __init__(self, input_size, hidden_sizes, output_size, use_sigmoid=False):
        # check if one or two hidden layers
        # note - hidden_sizes must be a list even if it is just one hidden layer.
        if len(hidden_sizes) > 1:
            self.two_hidden = True
        else:
            self.two_hidden = False
        # store the actual activation function directly in the object
        if use_sigmoid:
            # function for use in forward activation
            self.activation_function = self.sigmoid
            # function for use in reverse activation (includes derivative)
            self.activation_function_back = self.sigmoid_backprop
        else:
            # forward activiation
            self.activation_function = self.relu
            # reverse activation (includes derivative)
            self.activation_function_back = self.relu_backprop

        # if two hidden layers initialise 3 weight sets (w1i, w2j, w3k),
        # otherwise 2 (w1i, w2j)
        self.p = dict()
        if self.two_hidden:
            # Each neuron in previous layer has connections to each neuron in the next
            # Note that it is standard in Numpy neural networks
            # to represent the CURRENT layer in the rows and the PREVIOUS
            # layer in the columns. Hence by numpy notation, it looks
            # "back to front".
            # weight initialisation using sqrt taken from https://mlfromscratch.com/neural-network-tutorial/#/
            # It is Xavier Initilisation and designed to deal with the vanishing gradients
            # problem - i.e. losing gradient info as it is backpropograted through
            # multiple layers. (https://stats.stackexchange.com/questions/326710/why-is-weight-initialized-as-1-sqrt-of-hidden-nodes-in-neural-networks)
            self.p['w1'] = np.random.randn(hidden_sizes[0], input_size) * np.sqrt(1. / hidden_sizes[0])
            self.p['w2'] = np.random.randn(hidden_sizes[1], hidden_sizes[0]) * np.sqrt(1. / hidden_sizes[1])
            self.p['w3'] = np.random.randn(output_size, hidden_sizes[1]) * np.sqrt(1. / output_size)
        else:
            # Xavier initialisation but only needed for two weight sets
            self.p['w1'] = np.random.randn(hidden_sizes[0], input_size) * np.sqrt(1. / hidden_sizes[0])
            self.p['w2'] = np.random.randn(output_size, hidden_sizes[0]) * np.sqrt(1. / output_size)

    # https://numpy.org/doc/stable/reference/generated/numpy.where.html
    # relu is zero below 0, and just a straight line of gradient one above 0.
    def relu(self, x):
        return np.where(x > 0, x, 0)

    # https://stackoverflow.com/questions/46411180/implement-relu-derivative-in-python-numpy
    def relu_backprop(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # sigmoid derivative is well known and conveniently simple.
    # https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
    def sigmoid_backprop(self, x):
        return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)

    # https://machinelearningmastery.com/softmax-activation-function-with-python/
    def softmax(self, x):
        exp_x = np.exp(x - x.max())
        return exp_x / np.sum(exp_x, axis=0)

    # https://stackoverflow.com/questions/40575841/numpy-calculate-the-derivative-of-the-softmax-function/40576872
    def softmax_backprop(self, x):
        exp_x = np.exp(x - x.max())
        return exp_x / np.sum(exp_x, axis=0) * (1 - exp_x / np.sum(exp_x, axis=0))

    # forward activation is simple, just calculate the weights times the
    # the inputs, and sum - then put all through the activation.
    # With the matrix generated, move onto the next layer.
    # Then softmax at the output.
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

    # The more complicated. This was constructed with the help
    # of a couple of numpy-based examples online. One which did classification and
    # had a fixed number of hidden layers (1) and one which did regression and
    # allowed multiple hidden layers. Most examples seem to use the dictionary
    # approach.
    def activation_backprop(self, output_values, correct_outputs):
        w_delta = dict()
        if self.two_hidden:
            e3 = 2 * (correct_outputs - output_values
                         ) / correct_outputs.shape[0] * self.softmax_backprop(self.p['z3'])
            w_delta['w3'] = np.outer(e3, self.p['a2'])
            e2 = np.dot(self.p['w3'].T, e3
                           ) * self.activation_function_back(self.p['z2'])
            w_delta['w2'] = np.outer(e2, self.p['a1'])
            e1 = np.dot(self.p['w2'].T, e2
                           ) * self.activation_function_back(self.p['z1'])
            w_delta['w1'] = np.outer(e1, self.p['a0'])
        else:
            e2 = 2 * (correct_outputs - output_values
                         ) / correct_outputs.shape[0] * self.softmax_backprop(self.p['z2'])
            w_delta['w2'] = np.outer(e2, self.p['a1'])
            e1 = np.dot(self.p['w2'].T, e2
                           ) * self.activation_function_back(self.p['z1'])
            w_delta['w1'] = np.outer(e1, self.p['a0'])
        return w_delta

    # returns the accuracy of the NN predictions.
    def accuracy(self, X_test, y_test):
        y_hat_softmaxs = []
        for x, y in zip(X_test, y_test):
            # calualate neural net's best estimate
            y_hat = self.activation(x)
            # force to choose one class for both
            y_hat_softmax = np.argmax(y_hat)
            y_softmax = np.argmax(y)
            # compare estimate to ground truth
            y_hat_softmaxs.append(y_hat_softmax == y_softmax)
        # mean result of comparisons
        return np.mean(y_hat_softmaxs)

    # performs forward and back propogation
    # across the whole training set
    # for a given number of epochs
    # wth a chosen learning rate. Default learning rate
    # of 0.001 was most common online.
    def train(self, X_train, y_train, X_test, y_test,
              learning_rate=0.001, num_epochs=20):
        self.learning_rate = learning_rate
        epoch = 1
        while epoch <= num_epochs:
            # run through all training examples
            for x, y in zip(X_train, y_train):
                # calculate NN best estimate
                y_hat = self.activation(x)
                # find errors, backpropagate, and get resulting
                # weight changes based on gradient descent
                weight_deltas = self.activation_backprop(y, y_hat)
                # now update ALL weights in the network
                for weight, weight_delta in weight_deltas.items():
                    self.p[weight] -= self.learning_rate * weight_delta
            # calculate accuracy on test set
            acc = self.accuracy(X_test, y_test)
            print(f'Epoch {epoch} test acc {acc}')
            epoch += 1


neural_net = NumpyNeuralNet(input_size=784, hidden_sizes=[128, 64],
                            output_size=10, use_sigmoid=True)
# need this as was getting strange numpy error and online
# recommendation was to recast to float64. But did
# 32 and it worked.
X_train = np.asarray(X_train, dtype='float32')
y_train = np.asarray(y_train, dtype='float32')
X_test = np.asarray(X_test, dtype='float32')
y_test = np.asarray(y_test, dtype='float32')
neural_net.train(X_train, y_train, X_test, y_test, learning_rate=0.001, num_epochs=20)

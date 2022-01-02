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



X_train, y_train, X_test, y_test, class_names = get_mnist(redownload=REDOWNLOAD)
#print(X_train.shape)
X_train = np.reshape(X_train, (X_train.shape[0], 784))
X_test = np.reshape(X_test, (X_test.shape[0], 784))
#print(X_train.shape)

X_train, X_test = (X_train / 255).astype('float32'), (X_test / 255).astype('float32')
y_train, y_test = to_categorical(y_train), to_categorical(y_test)


class NumpyNeuralNet():
    def __init__(self, layers, use_sigmoid=False):
        if use_sigmoid:
            self.activation_function = self.sigmoid
            self.activation_function_back = self.sigmoid_backprop
        else:
            self.activation_function = self.relu
            self.activation_function_back = self.relu_backprop

        self.layers = layers

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
        # TO DO
        pass

    def activation_backprop(self, output_values, correct_outputs):
        # TO DO
        pass

    def train(self, X, y):
        # TO DO
        pass


neural_net = NumpyNeuralNet(layers=[784, 64, 10], use_sigmoid=True)


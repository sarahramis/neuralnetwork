from keras.utils.np_utils import to_categorical
import numpy as np
import tensorflow as tf
import pickle

np.random.seed(26)
# needs to be run with redownload set to True the first time
# this downloads mnist using tensorflow
# as saves it as a python pickle file
# then next time code is run, can set redownload to false
# and it will just load it from local file
def get_mnist(redownload=False):
    if redownload:
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
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        # store on local disk
        # https://ianlondon.github.io/blog/pickling-basics/
        with open('mnist_data.pickle', 'wb') as f:
            pickle.dump((train_images, train_labels,test_images, test_labels), f)
    else:
       # load the MNIST fashion from local disk (faster for development)
        with open('mnist_data.pickle', 'rb') as f:
            X_train, y_train, X_test, y_test = pickle.load(f)
    # tf doesn't provide the class names
    # https://www.tensorflow.org/tutorials/keras/classification
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return X_train, y_train, X_test, y_test, class_names



x_train, y_train, x_val, y_val, class_names = get_mnist()
#print(x_train.shape)
x_train = np.reshape(x_train, (x_train.shape[0], 784))
x_val = np.reshape(x_val, (x_val.shape[0], 784))
#print(x_train.shape)

x_train, x_val = (x_train / 255).astype('float32'), (x_val / 255).astype('float32')
y_train, y_val = to_categorical(y_train), to_categorical(y_val)

class NumpyNeuralNet():
    def __init__(self, layers):
        self.layers = layers
        # TO DO

    def activation(self, input_values):
        # TO DO
        pass

    def activation_backprop(self, output_values, correct_outputs):
        # TO DO
        pass

    def train(self, X, y):
        # TO DO
        pass


    def relu(x):
        # https://numpy.org/doc/stable/reference/generated/numpy.where.html
        return np.where(x > 0, x, 0)

    # https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
    def sigmoid_backprop(dA, x):
        sigmoid_x = sigmoid(x)
        # chain rule result
        return dA * sigmoid_x * (1 - sigmoid_x)

    # https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
    def relu_backprop(dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0;
        return dZ

    # https://numpy.org/doc/stable/reference/generated/numpy.reciprocal.html
    def sigmoid(x):
        return np.reciprocal(1 + np.exp(-1 * x))

    # https://machinelearningmastery.com/softmax-activation-function-with-python/
    def softmax(x):
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()


neural_net = NumpyNeuralNet([784, 64, 32, 10])


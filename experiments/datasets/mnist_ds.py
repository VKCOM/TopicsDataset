import tensorflow as tf
import pandas as pd
from keras.utils.np_utils import to_categorical

def get_mnist():
    (x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x = x.reshape(x.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x = x.astype('float32')
    x_test = x_test.astype('float32')
    x /= 255
    x_test /= 255
    return (x, y), (x_test, y_test)


def get_categorical_mnist():
    (x, y), (x_test, y_test) = get_mnist()
    y = to_categorical(y, 10)
    y_test = to_categorical(y_test, 10)
    return (x, y), (x_test, y_test)

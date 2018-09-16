import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

def load_mnist():

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train.reshape(-1,784) / 255.0, x_test.reshape(-1,784) / 255.0

    return x_train,y_train,x_test,y_test
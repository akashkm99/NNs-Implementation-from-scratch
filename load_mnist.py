import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

def load_mnist():

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train.reshape(-1,784) / 255.0, x_test.reshape(-1,784) / 255.0


    # len_train = y_train_.shape[0]
    # y_train = np.zeros([len_train,10])
    # y_train[np.arange(len_train),y_train_] = 1


    # len_test = y_test_.shape[0]
    # y_test = np.zeros([len_test,10])
    # y_test[np.arange(len_test),y_test_] = 1

    return x_train,y_train,x_test,y_test


# def hog():

    # x_train,y_train,x_test,y_test = load_mnist()
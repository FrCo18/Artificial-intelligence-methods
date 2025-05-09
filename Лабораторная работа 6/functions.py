import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def add_zero_feature(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))


def decode_y(y):
    num_labels = len(np.unique(y))
    Y = np.zeros((y.size, num_labels))
    for i in range(y.size):
        Y[i, y[i] - 1] = 1
    return Y


def rand_initialize_weights(L_in, L_out):
    epsilon = np.sqrt(6) / np.sqrt(L_in + L_out)
    return np.random.rand(L_out, L_in + 1) * 2 * epsilon - epsilon


def pack_params(Theta1, Theta2):
    return np.concatenate([Theta1.ravel(), Theta2.ravel()])


def unpack_params(nn_params, input_layer_size, hidden_layer_size, num_labels):
    Theta1_size = hidden_layer_size * (input_layer_size + 1)
    Theta1 = nn_params[:Theta1_size].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nn_params[Theta1_size:].reshape(num_labels, hidden_layer_size + 1)
    return Theta1, Theta2
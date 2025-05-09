import numpy as np
from functions import sigmoid


def sigmoid_gradient(z):
    g = sigmoid(z)
    derivative = g * (1 - g)
    return derivative
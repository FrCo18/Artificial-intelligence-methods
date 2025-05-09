import numpy as np
from functions import sigmoid, add_zero_feature, pack_params, unpack_params
from sigmoid_gradient import sigmoid_gradient

# Задание 5
def gradient_function(nn_params, input_layer_size, hidden_layer_size,
                      num_labels, X, Y, lambda_coef):
    Theta1, Theta2 = unpack_params(
        nn_params, input_layer_size, hidden_layer_size, num_labels)

    # количество примеров
    m = X.shape[0]

    # вычисление отклика нейронной сети
    A_1 = X
    Z_2 = np.dot(A_1, Theta1.T)
    A_2 = sigmoid(Z_2)
    A_2_with_bias = add_zero_feature(A_2)
    Z_3 = np.dot(A_2_with_bias, Theta2.T)
    A_3 = sigmoid(Z_3)

    # вычисление ошибок по нейронам
    DELTA_3 = A_3 - Y
    DELTA_2 = np.dot(DELTA_3, Theta2[:, 1:]) * sigmoid_gradient(Z_2)

    # вычисление частных производных
    Theta1_grad = np.dot(DELTA_2.T, A_1) / m
    Theta2_grad = np.dot(DELTA_3.T, A_2_with_bias) / m

    # добавление регуляризатора
    Theta1_grad[:, 1:] += (lambda_coef / m) * Theta1[:, 1:]
    Theta2_grad[:, 1:] += (lambda_coef / m) * Theta2[:, 1:]

    return pack_params(Theta1_grad, Theta2_grad)
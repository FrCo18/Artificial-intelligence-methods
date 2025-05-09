import numpy as np
from functions import sigmoid, add_zero_feature, unpack_params

# Задание 3
def cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels,
                  X, Y, lambda_coef):
    # распаковка параметров
    Theta1, Theta2 = unpack_params(
        nn_params, input_layer_size, hidden_layer_size, num_labels)

    # количество примеров
    m = X.shape[0]

    # вычисление отклика нейронной сети
    A_1 = X
    Z_2 = np.dot(A_1, Theta1.T)
    A_2 = sigmoid(Z_2)
    A_2 = add_zero_feature(A_2)
    Z_3 = np.dot(A_2, Theta2.T)
    A_3 = sigmoid(Z_3)
    H = A_3

    # вычисление ошибки
    J = (-1/m) * np.sum(Y * np.log(H) + (1 - Y) * np.log(1 - H + 1e-10))

    # вычисление регуляризатора
    reg_J = (lambda_coef/(2*m)) * (np.sum(Theta1[:, 1:]**2) + np.sum(Theta2[:, 1:]**2))

    J += reg_J

    return J
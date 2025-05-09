import numpy as np
from sigmoid import sigmoid

# 5. Вычисление отклика нейронной сети
def predict(Theta1, Theta2, X):
    # Добавляем bias-термин
    a1 = np.hstack((np.ones((len(X), 1)), X))

    # Прямой проход по первому слою
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((len(a2), 1)), a2))  # добавляем bias-термин ко второму слою

    # Прямой проход по второму слою
    z3 = np.dot(a2, Theta2.T)
    h_theta = sigmoid(z3)

    # Предсказываем класс
    predictions = np.argmax(h_theta, axis=1) + 1
    return predictions

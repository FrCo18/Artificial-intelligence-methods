import matplotlib.pyplot as plt
import numpy as np

from task3 import compute_cost

def gradient_descent(X, y, alpha, num_iterations):
    """
    Алгоритм градиентного спуска для нахождения коэффициентов регрессии

    Параметры:
    X - Матрица признаков (включая столбец единиц)
    y - Вектор целевой переменной
    alpha - Скорость обучения
    num_iterations - Максимальное число итераций
    """
    m = len(y)  # Число наблюдений
    n = X.shape[1]  # Число признаков
    theta = np.zeros(n)  # Начальная точка
    costs = []

    for _ in range(num_iterations):
        predictions = X.dot(theta)
        errors = predictions - y

        # Обновляем коэффициенты
        gradients = (alpha / m) * X.T.dot(errors)
        theta -= gradients

        # Сохраняем историю ошибок
        current_cost = compute_cost(X, y, theta)
        costs.append(current_cost)

    return theta, costs

# Пример использования функции
alpha = 0.02
num_iterations = 500

data = np.loadtxt("ex1data1.txt", delimiter=",")

# Разделение данных на две колонки: население и прибыль
X = data[:, 0]
y = data[:, 1]

# Преобразование одномерного массива в двумерный с добавлением единицы
X_ones = np.column_stack((np.ones(len(X)), X.reshape(-1, 1)))

# Теперь используем X_ones в качестве аргумента для градиентного спуска
final_theta, cost_history = gradient_descent(X_ones, y, alpha, num_iterations)
print(final_theta)
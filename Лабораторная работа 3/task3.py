import matplotlib.pyplot as plt
import numpy as np

def compute_cost(X, y, theta):
    """
    Вычисляем среднюю квадратичную ошибку для заданных параметров theta

    Параметры:
    X - Матрица признаков (включая столбец единиц)
    y - Вектор целевой переменной
    theta - Коэффициенты регрессии
    """
    m = len(y)  # Число наблюдений
    predictions = X.dot(theta)  # Гипотеза h(x)
    errors = predictions - y  # Ошибки
    cost = (errors ** 2).sum() / (2 * m)  # Средняя квадратичная ошибка
    return cost


# Загрузка данных из текстового файла
data = np.loadtxt("ex1data1.txt", delimiter=",")

# Разделение данных на две колонки: население и прибыль
X = data[:, 0]
y = data[:, 1]

X_ones = np.c_[np.ones((len(X), 1)), X.reshape(-1, 1)]  # Добавляем столбец единиц
theta_test = np.array([1, 2])
print(compute_cost(X_ones, y, theta_test))  # Должно вывести ~75.203
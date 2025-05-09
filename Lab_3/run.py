import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rc

data = np.matrix(np.loadtxt('ex1data1.txt',delimiter=','))

X=data[:,0]
y=data[:,1]

# Задание 1

font = {'family': 'Verdana', 'weight': 'normal'}
rc('font', **font)
plt.plot(X, y, 'b.')
plt.title('Зависимость прибыльности от численности')
plt.xlabel('Численность')
plt.ylabel('Прибыльность')
plt.grid()
plt.show()

m = X.shape[0] # количество элементов в X (количество городов)
X_ones = np.c_[np.ones((m, 1)), X] # добавляем единичный столбец к X
theta = np.matrix('[1; 2]') # коэффициенты theta представляют собой вектор-столбец из 2 элементов
h_x = X_ones * theta
# так можно вычислить значение гипотезы для всех городов сразу

def compute_cost(X, y, theta):
    m = len(y)
    h_x = np.dot(X, theta)
    a = np.power(h_x - y, 2)
    cost = np.sum(a) / (2 * m)
    return cost

print(compute_cost(X_ones,y,theta))


def gradient_descent(X, y, alpha, iterations):
    m, n = X.shape
    theta = np.zeros((n, 1))  # Инициализация вектора theta
    J_history = []  # Список для хранения значений J_theta на каждой итерации
    temp_theta=theta
    for i in range(iterations):
        h_x = np.dot(X, theta)  # Гипотеза h_theta(x)
        a = h_x - y
        gradient = np.dot(X.T, a) / m
        temp_theta -= alpha * gradient
        # Вычисление и сохранение значения J_theta
        cost = compute_cost(X,y,temp_theta)
        J_history.append(cost)

    return theta, J_history


alpha = 0.02
iterations = 500
theta, J_history = gradient_descent(X_ones, y, alpha, iterations)

# Вывод найденных значений theta
print("Оптимальные значения theta:", theta)

# Создание графика уменьшения ошибки J_theta
plt.plot(range(iterations), J_history, marker='o')
plt.title('Уменьшение ошибки J_theta с увеличением итераций')
plt.xlabel('Итерации')
plt.ylabel('J_theta')
plt.grid(True)
plt.show()


# Задание 6

font = {'family': 'Verdana', 'weight': 'normal'}
rc('font', **font)
plt.plot(X, y, 'b.')
s = np.arange(min(X), max(X))
plt.plot(s, theta[1]*s + theta[0], 'g--')
plt.title('Зависимость прибыльности от численности')
plt.xlabel('Численность')
plt.ylabel('Прибыльность')
plt.grid()
plt.show()

#Задание 5
# Значения численности новых городов
new_city1_population = 10000
new_city2_population = 20000

# Добавление единичной колонки и численности новых городов к матрице X
new_city_X = np.array([[1, new_city1_population],[1, new_city2_population]])

# Сделать предсказание для новых городов
predicted_profit1 = new_city_X[0].dot(theta)
predicted_profit2 = new_city_X[1].dot(theta)

# Вывести предсказанные прибыли
print("Предсказанная прибыли для первого нового города:", predicted_profit1)
print("Предсказанная прибыли для второго нового города:", predicted_profit2)

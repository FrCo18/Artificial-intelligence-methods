import matplotlib.pyplot as plt
import numpy as np

from task3 import compute_cost
from task4 import gradient_descent


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

# 5 Прогнозирование прибыли для новых городов
new_cities_population = np.array([10, 15]).reshape(-1, 1)
new_X = np.c_[np.ones(len(new_cities_population)), new_cities_population]
predicted_profits = new_X.dot(final_theta)
print(predicted_profits)

# 6
plt.figure(figsize=(8, 6))
plt.scatter(X, y)
plt.plot(X, final_theta[0] + final_theta[1]*X, color='green')
plt.xlabel("Население (тыс.)")
plt.ylabel("Прибыль ($ тыс.)")
plt.title("График зависимости прибыли от населения с линией регрессии")
plt.grid(True)
plt.show()

# task 7
def normalize_features(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    normalized_X = (X - mu) / sigma
    return normalized_X, mu, sigma

normalized_X, mu, sigma = normalize_features(X)

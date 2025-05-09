import numpy as np

def compute_cost(X, y, theta):
    m = len(y)
    h_x = np.dot(X, theta)
    a = np.power(h_x - y, 2)
    cost = np.sum(a) / (2 * m)
    return cost

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


#Task7
# Вычисляем средние значения и стандартные отклонения для каждого столбца

data = np.matrix(np.loadtxt('ex1data2.txt', delimiter=','))
X = data[:, 0]
y = data[:, 1]
means = np.mean(data, axis=0)
stds = np.std(data, axis=0)

# Нормализуем данные
normalized_data = (data - means) / stds

# Распечатываем средние значения и стандартные отклонения
print("Средние значения (means):", means)
print("Стандартные отклонения (stds):", stds)

# Печатаем нормализованные данные
print("Нормализованные данные:")
print(normalized_data)

# Для восстановления оригинальных данных используйте следующий код
# original_data = normalized_data * stds + means

#Task8
# Создаем матрицу X и вектор y
X = normalized_data[:, :-1]
y = normalized_data[:, -1]

# Параметры градиентного спуска
alpha = 0.02
iterations = 500

# Обучаем модель
theta_final, J_history = gradient_descent(X, y, alpha, iterations)

# Выводим итоговые значения theta
print("Итоговые значения theta(градиент):", theta_final)

# Делаем прогноз для двух неизвестных квартир
new_apartments = np.array([[1, 0.5], [1, -0.2]]) # Пример двух новых квартир
predictions = np.dot(new_apartments, theta_final)

# Выводим прогнозы
print("Прогнозы для двух новых квартир(градиент):")
print(predictions)

#Task9
# Создаем матрицу X и вектор y
X = normalized_data[:, :-1]
y = normalized_data[:, -1]

# Вычисляем коэффициенты theta с использованием метода наименьших квадратов (МНК)
X_transpose = X.T
XTX = np.dot(X_transpose, X)
XTy = np.dot(X_transpose, y)
theta_mnk = np.linalg.pinv(XTX).dot(XTy)

# Выводим итоговые значения theta
print("Итоговые значения theta (МНК):", theta_mnk)

# Делаем прогноз для двух новых квартир
new_apartments = np.array([[1, 0.5], [1, -0.2]]) # Пример двух новых квартир
predictions_mnk = np.dot(new_apartments, theta_mnk)

# Выводим прогнозы
print("Прогнозы для двух новых квартир (МНК):",predictions_mnk)

import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from displayData import displayData
from predict import predict

# Загружаем данные
data = io.loadmat('test_set.mat')
X = data['X'] # матрица признаков
y = data['y'].ravel() # целевые метки

# Загружаем веса
theta_data = io.loadmat('weights.mat')
Theta1 = theta_data['Theta1']
Theta2 = theta_data['Theta2']

m = len(y)
print(f"Количество образцов: {m}")

# 3. Отображение случайно выбранных цифр
plt.figure(figsize=(10,10))
random_indices = np.random.choice(range(m), size=100, replace=False)
example_digits = X[random_indices]
displayData(example_digits)
plt.show()

# 6. Оценка точности
predictions = predict(Theta1, Theta2, X)
accuracy = np.mean(predictions == y) * 100
print(f"Точность распознавания: {accuracy:.2f}%")

# 7. Демонстрация результатов на примерах
random_samples = np.random.randint(0, m, 5)
for idx in random_samples:
    sample_image = X[idx].reshape(20, 20)
    prediction = predict(Theta1, Theta2, X[[idx]])
    true_label = y[idx]
    print(f"Предсказанная цифра: {prediction}, Истинная цифра: {true_label}")

# 8. Поиск ошибок
errors_idx = np.where(predictions != y)[0][:100]
if errors_idx.size > 0:
    error_images = X[errors_idx]
    plt.figure(figsize=(10,10))
    displayData(error_images)
    plt.title("Примеры ошибок распознавания")
    plt.show()
else:
    print("Ошибок не обнаружено.")
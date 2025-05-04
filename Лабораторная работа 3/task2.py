import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных из текстового файла
data = np.loadtxt("ex1data1.txt", delimiter=",")

# Разделение данных на две колонки: население и прибыль
X = data[:, 0]
y = data[:, 1]

plt.figure(figsize=(8, 6))
plt.scatter(X, y)
plt.xlabel("Население (тыс.)")
plt.ylabel("Прибыль ($ тыс.)")
plt.title("График зависимости прибыли от населения")
plt.grid(True)
plt.show()
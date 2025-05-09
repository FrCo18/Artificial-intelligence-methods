import numpy as np

# 4. Реализация сигмоидной функции
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

import numpy as np
from scipy.io import loadmat
import pandas as pd

# Загрузка данных из текстового файла
data = np.loadtxt("ex1data1.txt", delimiter=",")

# Разделение данных на две колонки: население и прибыль
X = data[:, 0]
y = data[:, 1]
import numpy as np


# Функция для чтения данных из файла
def read_input_from_file(filename):
    with open(filename, 'r') as f:
        data = f.read().splitlines()  # Разделяем файл на строки

    # Первое значение — количество городов
    N = int(float(data[0]))

    # Остальные строки содержат расстояния
    matrix_data = []  # Будущая матрица расстояний

    for line in data[1:]:
        numbers = line.split()  # Разделение строки на числа
        row = list(map(float, numbers))  # Преобразование чисел в float
        matrix_data.extend(row)

    # Конвертирование в массив NumPy и формирование квадратной матрицы
    adjacency_matrix = np.array(matrix_data).reshape((N, N))

    return N, adjacency_matrix


# Использование функции
filename = "var2.txt"
N, matrix = read_input_from_file(filename)

# Выводы
print("Количество городов:", N)
print("Матрица расстояний:")
print(matrix)

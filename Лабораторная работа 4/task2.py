import numpy as np

from task1 import read_input_from_file


filename = "var2.txt"
# Матрица расстояний
N, matrix = read_input_from_file(filename)


# Функция для расчета общей стоимости маршрута
def calculate_total_cost(route, distances):
    """
    Рассчитывает общую стоимость маршрута коммивояжера.

    Параметры:
    route (list of int): Список индексов городов (нумерация начинается с 0).
    distances (numpy array): Матрица расстояний между городами.

    Возвращает:
    float: Общая стоимость маршрута.
    """
    # Добавляем возвращение в исходный город
    full_route = route + [route[0]]
    cost = 0
    for i in range(len(full_route) - 1):
        from_city = full_route[i]
        to_city = full_route[i + 1]
        cost += distances[from_city][to_city]
    return cost


# Пример маршрута
example_route = [0, 3, 1, 2, 4, 6, 5]

# Расчёт стоимости
total_cost = calculate_total_cost(example_route, matrix)
print(f"Общая стоимость маршрута: {total_cost:.1f}")

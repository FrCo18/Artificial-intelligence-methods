import random


def greedy_crossover(parent1, parent2):
    """
    Осуществляет жадный кроссинговер для задачи коммивояжера.

    Параметры:
    parent1 (list of int): Первый родитель (маршрут).
    parent2 (list of int): Второй родитель (маршрут).

    Возвращает:
    list of int: Потомство (новый маршрут).
    """

    def find_next_closest(city, used_cities, other_parent):
        """Находит ближайший доступный город"""
        min_dist = float('inf')
        next_city = None

        for c in other_parent:
            dist = abs(c - city)  # Простое расстояние между номерами городов

            if c not in used_cities and dist < min_dist:
                min_dist = dist
                next_city = c

        return next_city

    child = []
    current_parent = parent1
    start_point = random.choice(current_parent)
    child.append(start_point)
    used_cities = set(child)

    while len(child) < len(parent1):
        current_city = child[-1]
        next_city = find_next_closest(current_city, used_cities, current_parent)

        if next_city is None:
            # Если больше нет ближайших доступных городов в текущем родителе
            # Переключаемся на второго родителя
            current_parent = parent2 if current_parent == parent1 else parent1
            next_city = find_next_closest(current_city, used_cities, current_parent)

        child.append(next_city)
        used_cities.add(next_city)

    return child


# Пример использования
parent1 = [0, 3, 1, 2, 4, 6, 5]
parent2 = [6, 3, 5, 4, 2, 1, 0]
child = greedy_crossover(parent1, parent2)
print("Родитель 1:", parent1)
print("Родитель 2:", parent2)
print("Потомок:", child)

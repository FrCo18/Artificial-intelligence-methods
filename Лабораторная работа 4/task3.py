import random


def mutation(route):
    """
    Применяет операцию мутации к маршруту.

    Параметры:
    route (list of int): Исходный маршрут.

    Возвращает:
    list of int: Новый мутированный маршрут.
    """
    if len(route) <= 2:
        return route  # Нет смысла менять маршрут из одного-двух элементов

    # Выбор случайного индекса для перестановки
    idx_to_move = random.randint(0, len(route) - 1)
    city_to_move = route[idx_to_move]

    # Удаляем город из старого места
    del route[idx_to_move]

    # Выбор нового случайного места
    new_idx = random.randint(0, len(route))  # Возможно добавить элемент в конец тоже

    # Вставляем город обратно в новый индекс
    route.insert(new_idx, city_to_move)

    return route


# Пример использования
original_route = [0, 3, 1, 2, 4, 6, 5]
mutated_route = mutation(original_route[:])  # Копируем оригинальный маршрут, чтобы сохранить оригинал
print("Оригинальный маршрут:", original_route)
print("Мутировавший маршрут:", mutated_route)

import random


def generate_random_path(num_cities):
    """
    Генерирует случайный маршрут, включающий все города.

    Параметры:
    num_cities (int): Количество городов.

    Возвращает:
    list of int: Случайный маршрут.
    """
    cities = list(range(num_cities))  # Список номеров городов
    random.shuffle(cities)  # Перемешиваем города случайным образом
    return cities


def create_initial_population(population_size, num_cities):
    """
    Создает начальную популяцию.

    Параметры:
    population_size (int): Размер популяции (количество особей).
    num_cities (int): Количество городов.

    Возвращает:
    list of lists: Начальная популяция.
    """
    population = []
    for _ in range(population_size):
        population.append(generate_random_path(num_cities))
    return population


# Пример использования
population_size = 10
num_cities = 7

initial_population = create_initial_population(population_size, num_cities)
for i, individual in enumerate(initial_population):
    print(f"Поколение {i}: {individual}")

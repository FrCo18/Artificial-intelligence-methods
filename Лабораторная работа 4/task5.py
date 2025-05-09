import random

from task1 import read_input_from_file
from task3 import mutation
from task4 import greedy_crossover


# Фитнес-функция: подсчёт стоимости маршрута
def fitness_function(route, distances):
    total_cost = 0
    n = len(route)
    for i in range(n):
        # Суммируем расстояния между всеми соседними городами
        total_cost += distances[route[i], route[(i+1) % n]]
    return total_cost

# Отбор элитных особей
def select_elites(population, k, distances_matrix):
    sorted_pop = sorted(population, key=lambda r: fitness_function(r, distances_matrix))
    return sorted_pop[:k]

# Главная функция генетического алгоритма
def genetic_algorithm(num_generations, population_size, mutation_rate, elitism_rate):
    filename = "var2.txt"
    # Матрица расстояний
    N, distances_matrix = read_input_from_file(filename)

    num_cities = len(distances_matrix)
    population = [random.sample(range(num_cities), num_cities) for _ in range(population_size)]

    best_routes = []

    for gen in range(num_generations):
        # Элитизм: сохранение лучших особей
        elite_count = max(int(elitism_rate * population_size), 1)
        elites = select_elites(population, elite_count, distances_matrix)

        # Создание потомков
        offspring = []
        for _ in range((population_size - elite_count)):
            parents = random.sample(population, 2)
            child = greedy_crossover(*parents)
            offspring.append(child)

        # Приложение мутаций
        for route in offspring:
            if random.random() < mutation_rate:
                mutation(route)

        # Формирование новой популяции
        population = elites + offspring

        # Лучший маршрут текущего поколения
        best_routes.append(min(population, key=lambda r: fitness_function(r, distances_matrix)))

    # Поиск оптимального маршрута
    final_best_route = min(best_routes, key=lambda r: fitness_function(r, distances_matrix))
    return final_best_route, fitness_function(final_best_route, distances_matrix)


# Экспериментальные настройки
num_generations = 100      # Количество поколений
population_size = 50       # Размер популяции
mutation_rate = 0.1        # Вероятность мутации
elitism_rate = 0.1         # Процент элитных особей

# Запуск генетического алгоритма
best_route, best_fitness = genetic_algorithm(
    num_generations=num_generations,
    population_size=population_size,
    mutation_rate=mutation_rate,
    elitism_rate=elitism_rate
)

# Печать итогового результата
print()
print("Task 5:")
print("Лучший маршрут:", best_route)
print("Оптимальная стоимость маршрута:", best_fitness)


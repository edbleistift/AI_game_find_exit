import numpy as np
import pygame
import random

# Параметры лабиринта и обучения
width, height = 10, 10  # Размеры лабиринта
learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0  # Начальное значение epsilon
epsilon_decay = 0.995  # Как быстро уменьшается epsilon
epsilon_min = 0.1  # Минимальное значение epsilon

# Создаем лабиринт (0 — путь, 1 — стена, S — старт, E — выход)
maze = np.array([
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
    [1, 1, 0, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 1, 1, 0, 0]
])

start = (9, 0)
end = (8, 9)

# Инициализируем Q-таблицу
q_table = np.zeros((width, height, 4))  # Четыре действия: вверх, вниз, влево, вправо
actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # (x, y) шаги

# Функции для выбора действия и обновления Q-таблицы
def choose_action(state):
    global epsilon
    if random.uniform(0, 1) < epsilon:  # ε-greedy
        return random.randint(0, 3)  # Случайное действие
    else:
        return np.argmax(q_table[state[0], state[1]])  # Оптимальное действие

def update_q_table(state, action, reward, next_state):
    best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
    td_target = reward + discount_factor * q_table[next_state[0], next_state[1], best_next_action]
    td_error = td_target - q_table[state[0], state[1], action]
    q_table[state[0], state[1], action] += learning_rate * td_error

# Параметры визуализации
cell_size = 30
fps = 1000

# Инициализация Pygame
pygame.init()
screen = pygame.display.set_mode((width * cell_size, height * cell_size))
pygame.display.set_caption("Обучение ИИ в лабиринте")

# Цвета для визуализации
colors = {
    "path": (255, 255, 255),    # Белый цвет для пути
    "wall": (123, 111, 0),      # Коричневый цвет для стен
    "start": (0, 255, 0),       # Зеленый цвет для старта
    "end": (255, 0, 0),         # Красный цвет для выхода
    "agent": (0, 0, 255)        # Синий цвет для агента
}

# Создание матрицы для отображения стен
walls_matrix = np.array([
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
    [1, 1, 0, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 1, 1, 0, 0]
])

# Функция для отрисовки лабиринта и агента
def draw_maze(agent_position, episode, steps):
    screen.fill(colors["path"])  # Задаем фон всего экрана как белый (цвет пути)
    
    # Рисуем путь, старт и выход
    for y in range(height):
        for x in range(width):
            if (y, x) == start:
                pygame.draw.rect(screen, colors["start"], pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size))
            elif (y, x) == end:
                pygame.draw.rect(screen, colors["end"], pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size))

    # Отрисовка стен из второй матрицы
    for y in range(height):
        for x in range(width):
            if walls_matrix[y, x] == 1:  # Если в стенах есть стена
                pygame.draw.rect(screen, colors["wall"], pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size))

    # Отрисовка агента
    pygame.draw.rect(screen, colors["agent"], pygame.Rect(agent_position[1] * cell_size, agent_position[0] * cell_size, cell_size, cell_size))

    # Отрисовка сетки
    for y in range(height + 1):  # Рисуем горизонтальные линии
        pygame.draw.line(screen, (0, 0, 0), (0, y * cell_size), (width * cell_size, y * cell_size), 2)
    for x in range(width + 1):  # Рисуем вертикальные линии
        pygame.draw.line(screen, (0, 0, 0), (x * cell_size, 0), (x * cell_size, height * cell_size), 2)

    # Отображаем эпизод и количество шагов
    font = pygame.font.Font(None, 36)
    text = font.render(f"Episode: {episode}  Steps: {steps}", True, (0, 0, 0))
    screen.blit(text, (10, height * cell_size - 40))

    pygame.display.flip()

# Основной цикл обучения
for episode in range(500):  # Количество эпизодов обучения
    state = start
    steps = 0
    for step in range(100):  # Ограничение на количество шагов
        action = choose_action(state)
        next_state = (state[0] + actions[action][0], state[1] + actions[action][1])

        # Если агент выходит за пределы лабиринта, остаемся в том же месте
        if not (0 <= next_state[0] < height and 0 <= next_state[1] < width) or maze[next_state[0], next_state[1]] == 1:
            next_state = state  # Остаться на месте

        # Награды: за достижение выхода +100, за шаг -1
        if next_state == end:
            reward = 100  # Награда за выход
        else:
            reward = -1  # Штраф за каждый шаг

        update_q_table(state, action, reward, next_state)

        state = next_state
        steps += 1

        # Отображаем лабиринт
        draw_maze(state, episode, steps)

        # Пауза между шагами
        pygame.time.Clock().tick(fps)

        # Если агент достиг выхода, выходим из цикла
        if state == end:
            print(f"Agent reached the exit in episode {episode}!")
            break

    # Уменьшаем epsilon после каждого эпизода
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

# После обучения: агент идет по лабиринту, следуя обученной стратегии
while True:
    state = start
    while state != end:
        action = choose_action(state)
        next_state = (state[0] + actions[action][0], state[1] + actions[action][1])

        # Если агент выходит за пределы лабиринта, остаемся в том же месте
        if not (0 <= next_state[0] < height and 0 <= next_state[1] < width) or maze[next_state[0], next_state[1]] == 1:
            next_state = state  # Остаться на месте

        state = next_state
        draw_maze(state, 0, 0)  # Отображаем лабиринт
        pygame.time.Clock().tick(fps)  # Задержка для визуализации

    # После достижения выхода, агент продолжит до следующего цикла
    print("Agent reached the exit!")
    
# Завершаем Pygame
pygame.quit()

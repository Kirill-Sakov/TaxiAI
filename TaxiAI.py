import gym
import numpy as np
import random
from time import sleep
from IPython.display import clear_output
import os

# Инициализируем среду
env = gym.make("Taxi-v3").env

# Инициализируем произвольные значения
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Гиперпараметры
alpha = 0.1  # темп обучения
gamma = 0.6  # важность предсказанных наград
epsilon = 0.1  # Исследуй-эксплуатируй

game_frames = []  # кадры

# Обучаем агента
for i in range(1, 100001):
    state = env.reset()  # обновляем среду

    # Инициализируем переменные
    epochs, penalties, reward, = 0, 0, 0
    done = False

    # Пока не доставил пассажира
    while not done:
        # В зависимости от эпсилон выбираем либо уже извстные значения
        # Либо экспериментируем с новыми
        if random.uniform(0, 1) < epsilon:
            # Проверяем пространство действий
            action = env.action_space.sample()
        else:
            # Проверяем изученные значения
            action = np.argmax(q_table[state])

        # Совершаем шаг и записываем
        # 1) следующее состояние
        # 2) Награду
        # 3) Удалось ли успешно высадить пассажира
        next_state, reward, done, info = env.step(action)

        # Записываем старое Q-значение, которое находилось на том действии которое выбрали
        old_value = q_table[state, action]
        # Выбираем максимальное известное Q-начение для следующего действия в следующей ситуации
        next_max = np.max(q_table[next_state])

        # Обновляем новое значение
        new_value = (1 - alpha) * old_value + alpha * \
            (reward + gamma * next_max)

        # Заносим в Q-таблицу новое Q-значение для данного действия в данной ситуации
        q_table[state, action] = new_value

        # Обновляем состояние
        state = next_state

        if i % 100 == 0:
            # Каждый отображенный кадр помещаем в словарь для анимации
            game_frames.append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
            }
            )

        epochs += 1

    if i % 100 == 0:
        clear_output(wait=True)
        print("Episode: ", i)

print("Training finished.")


# Выводим все возможные действия, состояния, вознаграждения
def frames(game_frames):
    for i, frame in enumerate(game_frames):
        if i > 0:
            os.system('cls' if os.name == 'nt' else 'clear')
            clear_output(wait=True)
            print(frame['frame'])
            print(f"Timestep: {i + 1}")
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Reward: {frame['reward']}")
            sleep(.2)


frames(game_frames)
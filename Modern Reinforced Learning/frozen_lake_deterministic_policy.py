import gym
import numpy as np
import matplotlib.pyplot as plt
import os

env = gym.make("FrozenLake-v0")
env.reset()
episodes = 1_000_000


epsilon = 0.95
gamma = 0.9
epsilon_decay = 0.9999995
learning_rate = 0.001
scores = []
win_pcts = []

q_table = np.zeros(shape=(16, 4))
win = 0
win2 = 0
for episode in range(episodes):
    done = False
    state = env.reset()
    score = 0
    # os.system("cls")
    # print(win)
    # print(episode)
    # print(epsilon)
    while not done:

        if np.random.rand() < epsilon:
            action = np.random.choice(4, 1)[0]

        else:
            action = np.argmax(q_table[state])

        new_state, reward, done, _ = env.step(action)

        # Q update

        Q_now = q_table[(state,action)]
        future_Q = learning_rate * (reward + gamma * np.max(q_table[new_state]) - Q_now)
        q_table[(state, action)] = Q_now + future_Q

        if epsilon > 0.01:
            epsilon *= epsilon_decay
        else:
            epsilon = 0.01

        score += reward
        state = new_state
    scores.append(score)

    if episode == 500_000:
        epsilon = 0.95

    if episode % 100 == 0:
        win_pct = np.mean(scores[-100:])
        win_pcts.append(win_pct)
    if episode % 100_000 == 0:
        os.system("cls")
        print(episode)


plt.plot(win_pcts)
plt.show()

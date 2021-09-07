import numpy as np
import gym
import matplotlib.pyplot as plt
from Agent import Agent

if __name__ == '__main__':
    # instantiate our environment
    env = gym.make('FrozenLake-v0')
    env.render()

    # object agent
    agent = Agent(lr=0.001, gamma=0.9, n_actions=4, n_states=16, eps_start=1, eps_end=0.01, eps_dec=0.9999995)

    # game
    scores = []
    win_pcts = []
    n_games = 500000

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            # choose action
            action = agent.choose_action(observation)
            # new values after taking the action
            observation_, reward, done, info = env.step(action)
            # learn from new observation
            agent.learn(state=observation, action=action, reward=reward, state_=observation_)
            # update score with reward
            score += reward
            # update observation
            observation = observation_

        # append score to the list at the end of episode
        scores.append(score)

        # for every 100 games, calculate the win %
        if i % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pcts.append(win_pct)
            # for every 1000 games, print the debug information
            if i % 1000 == 0:
                print(f"Episode: {i}\tWin % = {round(win_pct, 2)}\t Epsilon = {round(agent.epsilon, 2)}")

    # plot win %
    plt.plot(win_pcts)
    plt.show()
import gym
import time
import cv2
import numpy as np
from image_preprocessing import preprocess, plot_learning_curve
from Dueling_Agent import DQNAgent

# env = gym.make('MsPacman-v0')
# env.reset()
#
# for i in range(1):
#     obs, reward, done, info = env.step(0)
#
#
# print(obs.shape)
# obs = preprocess(obs)
# print(obs.shape)
# print(type(obs))
#
# print(np.amin(obs))

if __name__ == "__main__":
    env = gym.make('MsPacman-v0')
    obs = env.reset()
    obs = preprocess(obs)

    best_score = -np.inf
    episodes = 500
    load_checkpoint = False

    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=(1,88,80),
                     n_actions=env.action_space.n, mem_size=30000, eps_min=0.1,
                     batch_size=32, replace=1000, eps_decay=1e-5,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='MsPacman-v0')

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' \
            + str(episodes) + 'games'
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    scores, eps_history, steps_array = [], [], []
    render = False
    for episode in range(episodes):
        done = False
        observation = env.reset()
        observation = preprocess(observation)

        score = 0

        if episode == 248:
            render = True

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            # if reward > 0.0:
            #     print("reward", reward)
            observation_ = preprocess(observation_)
            score += reward
            if not load_checkpoint:
                agent.store_memory(observation, action,
                                       reward, observation_, done)
                agent.learn()
            # if render:
            #     env.render()

            observation = observation_
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', episode, 'score: ', score,
                  ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
                  'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score



        eps_history.append(agent.epsilon)

    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
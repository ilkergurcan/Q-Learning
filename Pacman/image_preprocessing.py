import cv2
import gym
import numpy as np
import matplotlib.pyplot as plt


class Preprocessing(gym.Wrapper):
    def __init__(self, env=None):
        super(Preprocessing, self).__init__(env)


def makeGrayscale(obs):
    new_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY, dst=1)
    #print("sa",new_obs.shape)
    return new_obs

def crop_and_resize(obs):
    cropped = obs[1:180, 4:156]

    new_obs = cv2.resize(cropped, (88, 80))
    #new_obs = np.reshape(new_obs, (88,80))
    #print("new", new_obs.shape)
    return new_obs

def normalize(obs):
    obs = cv2.normalize(obs, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    obs = np.reshape(obs, (1,88,80))

    #print("obs", obs.shape)
    return obs
def preprocess(obs):
    observation = makeGrayscale(obs)
    observation = crop_and_resize(observation)
    observation = normalize(observation)

    return observation

def plot_learning_curve(x, scores, epsilons, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig(filename)
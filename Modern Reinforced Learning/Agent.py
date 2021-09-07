# import libraries
import numpy as np


# Agent Class
class Agent:
    # initializing: learning rate lr, discount factor gamma, number of actions,
    #               number of states, epsilon start, epsilon end, epsilon
    #               decrement factor
    def __init__(self, lr, gamma, n_actions, n_states, eps_start, eps_end, eps_dec):
        # save all those parameters as member variables of the class
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        # epsilon will decrement from eps start to eps end
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec

        # initialize the Q Table as empty dictionary
        self.Q = {}

        # function to initialize the items in Q Table
        self.init_Q()

    # function body for init Q
    def init_Q(self):
        # set all state-action value with arbitrary value (here, zero)
        # but for terminal state, it must be zero
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q[(state, action)] = 0.0

    # function to choose action based on state
    def choose_action(self, state):
        # follows epsilon-greedy strategy
        # if random number < eps, go for exploration else go with exploitation
        if np.random.random() < self.epsilon:
            action = np.random.choice([i for i in range(self.n_actions)])
        else:
            # extract the action values available for the state
            action_vals = np.array([self.Q[(state, a)] for a in range(self.n_actions)])
            # choose action whose action value is maximum
            action = np.argmax(action_vals)
        return action

    # function to decrement epsilon
    def decrement_epsilon(self):
        # any decrement method can be used, here it is linear
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min

    # learning function
    # inputs: state, action, reward, and new state
    def learn(self, state, action, reward, state_):
        # action values for the new state (state_)
        action_vals = np.array([self.Q[(state, a)] for a in range(self.n_actions)])
        # max value action
        a_max = np.argmax(action_vals)

        # update policy (Q Table)
        self.Q[(state, action)] += self.lr * (reward
                                              + self.gamma * self.Q[(state_, a_max)]
                                              - self.Q[(state, action)])

        # decrement epsilon
        self.decrement_epsilon()
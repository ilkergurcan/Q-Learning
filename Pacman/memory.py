import numpy as np

class MemoryBuffer(object):
    def __init__(self, max_size , n_actions, input_shape=(88,80)):
        self.input_shp = (88,80,1)
        self.mem_size = 30000
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, 1,88, 80),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, 1,88, 80),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.done_memory = np.zeros(self.mem_size, dtype=np.bool)


    def store_memory(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size
        #print("state",state.shape)

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.done_memory[index] = done
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.done_memory[batch]

        return states, actions, rewards, states_, terminal

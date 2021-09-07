import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DuelingDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv = nn.Sequential(nn.Conv2d(1, 32, 8, stride=4),
                      nn.ReLU(),
                      nn.Conv2d(32, 64, 4, stride=2),
                      nn.ReLU(),
                      nn.Conv2d(64, 64, 3, stride=1),
                      nn.ReLU(),
        )

        self.fc1 = nn.Linear(2688, 512)
        #self.fc2 = nn.Linear(512, n_actions)

        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        conv_layer = self.conv(state)

        flatten = conv_layer.view(conv_layer.size(0), -1)

        hidden = F.relu(self.fc1(flatten))

        V = self.V(hidden)
        A = self.A(hidden)
        return V, A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
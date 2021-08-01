import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Util funtion
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

# Params
FC1_UNITS = 256         # Number of nodes in the first hidden layer
FC2_UNITS = 256         # Number of nodes in the second hidden layer

class Critic(nn.Module):
    """ Critic (Value) model for DDPG """

    def __init__(self, state_size, action_size, seed=0):
        """Constructor for Critic model to initialize states, actions and random seed
        Args:
            state_size:  number of states
            action_size: number of actions
            seed: rng seed value
        """
        super(Critic, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size*2, FC1_UNITS)              # First Layer
        self.bn1 = nn.BatchNorm1d(FC1_UNITS)                         # First layer normalization
        self.fc2 = nn.Linear(FC1_UNITS+(action_size*2), FC2_UNITS)  # Second Layer
        self.bn2 = nn.BatchNorm1d(FC2_UNITS)                        # Second layer normalization
        self.fc3 = nn.Linear(FC2_UNITS, 1)                          # Third Layer

        # initialize layers
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Network of state to action values
        Args:
            state: state to map to an action
        Returns:
            mapped state to action values
        """
        xs = F.relu(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)
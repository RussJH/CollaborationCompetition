import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# util function
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

FC1_UNITS = 256         # Nodes in the first hidden layer
FC2_UNITS = 256         # Nodes in the second hidden layer

class Actor(nn.Module):
    """ Actor (Policy) model for DDPG """

    def __init__(self, state_size, action_size, seed=0):
        """Constructor for Actor model to initialize states, actions and random seed
        Args:
            state_size:  number of states
            action_size: number of actions
            seed: rng seed value
        """

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size*2, FC1_UNITS)   # First Layer
        self.bn1 = nn.BatchNorm1d(FC1_UNITS)            # First layer normalization
        self.fc2 = nn.Linear(FC1_UNITS, FC2_UNITS)      # Second Layer
        self.bn2 = nn.BatchNorm1d(FC2_UNITS)            # Second layer normalization
        self.fc3 = nn.Linear(FC2_UNITS, action_size)    # Third Layer
        
        # Initialize layers
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Network of state to action values
        Args:
            state: state to map to an action
        Returns:
            mapped state to action values
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
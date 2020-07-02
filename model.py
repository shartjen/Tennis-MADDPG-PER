import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, hidden_size, action_size, seed, dropout): #, fc1_units=24, fc2_units=48):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_layer = nn.Linear(state_size, hidden_size)
        self.batchnorm_layer = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.dropout_layer = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(int(hidden_size/2), action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.input_layer(state))
        # x = F.relu(self.batchnorm_layer(x))
        x = F.relu(self.fc1(x))
        x = self.dropout_layer(x)
        return torch.tanh(self.output_layer(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, hidden_size, action_size, seed, dropout): #, fcs1_units=24, fc2_units=48):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size :  Number of nodes in the first hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_layer = nn.Linear(state_size, hidden_size)
        self.batchnorm_layer = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size + action_size, int(hidden_size/2))
        self.dropout_layer = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(int(hidden_size/2), 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = (self.input_layer(state))
        x = F.relu(self.batchnorm_layer(x))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout_layer(x)
        return self.output_layer(x)

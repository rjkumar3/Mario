import torch
from torch import nn
import numpy as np


# Class for our Neural network which will consist of 3 Convolutional layers
# and 2 linear layers

class AgentNN(nn.Module):
    def __init__(self, input_shape, n_actions, freeze=False):
        super().__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        # Linear layers
        self.network = nn.Sequential(
            self.conv_layers,
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        if freeze:
            self._freeze()

        # command that allows the user to specify if the network will be run on a GPU or default to CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

# Forward pass
    def forward(self, x):
        return self.network(x)


# Performs dummy forward pass through the convolutional layers
# to determine the number of neurons we need in out initial linear layer
    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        # np.prod returns the product of array elements over a given axis
        return int(np.prod(o.size()))

# Prevents pytorch from calculating gradients which we need for our target network
# We only use the target network to calculate the correct values we want out online network to predict
    def _freeze(self):        
        for p in self.network.parameters():
            p.requires_grad = False
    
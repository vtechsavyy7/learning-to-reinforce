from torch import nn
import numpy as np
import torch


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        """A convolutional neural network architecture for the DQN agent, based on the original paper by Mnih et al. (2015)
        
        Args:
            input_shape (tuple): The shape of the input state representation (e.g., (4, 84, 84) for a stack of 4 grayscale frames)
            num_actions (int): The number of possible actions in the environment (e.g., 6 for Space Invaders)
        """
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_dim = self._get_conv_out_dim(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out_dim(self, shape):
        """Calculate the output dimension of the convolutional layers given the input shape, by passing a dummy tensor through the conv layers
        
        Args:
            shape (tuple): The shape of the input state representation (e.g., (4, 84, 84))
        
        Returns:
            int: The output dimension of the convolutional layers
        """
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """Forward pass through the network to compute Q-values for each action given the input state representation.
        
        Args:
            x (torch.Tensor): The input state representation as a tensor of shape (batch_size, channels, height, width) (e.g., (32, 4, 84, 84))
            
        Returns:
            torch.Tensor: The output Q-values for each action, as a tensor of shape (batch_size, num_actions) (e.g., (32, 6))
        """
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

# Class to represent the neural network for the actor in the policy gradient method. 
# It is a convolutional neural network that takes in the state of the environment and outputs the action probabilities.
import torch
from torch import nn

class ActorCNN(nn.Module):
    def __init__(self, input_shape, action_size):
        """Initialize the ActorCNN.
        
        Args:
            input_shape (tuple): shape of the input state (e.g., (4, 84, 84) for a stack of 4 grayscale frames)
            action_size (int): number of possible actions"""
        super(ActorCNN, self).__init__()

        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Define the fully connected layers
        conv_out_shape = self._get_conv_output(input_shape)  # Calculate the output size of the convolutional layers

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_shape, 512),
            nn.ReLU(), 
            nn.Linear(512, action_size),
            nn.Softmax(dim=-1)  # Output action probabilities
        )

    def _get_conv_output(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv_layers(o)
        return int(torch.prod(torch.tensor(o.size())))

    def forward(self, x):
        """Forward pass through the network.
        Args:
            x (torch.Tensor): input state [batch_size, channels, height, width]
        Returns:
            torch.distributions.Categorical: a categorical distribution over the action probabilities"""
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        # Create a categorical distribution over the action probabilities
        # This is useful for sampling actions during training and for calculating the log probabilities for the policy gradient update.
        action_probs = torch.distributions.Categorical(x)

        return action_probs
    
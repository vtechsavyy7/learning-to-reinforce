# A class to represent an agent that uses the REINFORCE policy gradient algorithm to learn a policy for selecting actions in an environment.
# The agent uses an ActorCNN neural network to represent the policy, which takes in the state of the environment and outputs a probability distribution over the possible actions. 
# The agent also uses an optimizer to update the weights of the ActorCNN based on the policy gradient loss, which is calculated using the log probabilities of the actions taken and the discounted rewards received.
from actor_cnn import ActorCNN
import torch
from torch import optim
import numpy as np

class REINFORCEAgent:
    def __init__(self, input_shape, action_size, device, learning_rate, gamma):
        """Initialize the REINFORCE agent with the given parameters
        
        Args:
            input_shape (tuple): The shape of the input state representation (e.g., (4, 84, 84) for a stack of 4 grayscale frames)
            action_size (int): The number of possible actions in the environment (e.g., 6 for Space Invaders)
            device (torch.device): The device to store the tensors on
            learning_rate (float): The learning rate for the optimizer
            gamma (float): The discount factor for future rewards
        """
        self.device = device
        self.gamma = gamma

        self.policy_net = ActorCNN(input_shape, action_size).to(device)
        self.policy_net.train()  # Set the policy network to training mode
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # A list to store the log probabilities of the actions taken during an episode, used for calculating the policy gradient loss at the end of the episode
        self.log_probs = [] 
        # A list to store the rewards received during an episode, used for calculating the discounted returns for the policy gradient update at the end of the episode
        self.rewards = []
        # A list to store the done masks for each time step in the episode, used for calculating the discounted returns for the policy gradient update at the end of the episode
        self.dones = []


    def select_action(self, state):
        """Select an action based on the current state and the policy network's action probabilities
        
        Args:
            state (numpy array): The current state representation (e.g., a queue of 4 preprocessed frames)
        
        Returns:
            int: The index of the selected action
            torch.Tensor: The log probability of the selected action, used for calculating the policy gradient loss during training
        """
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32, device=self.device)  # Add batch dimension

        # Move the policy network to evaluation mode to disable dropout and batch normalization during action selection
        # NOTE: Make sure that the policy network is in training mode during training, 
        # and set to evaluation mode during action selection in inference to ensure consistent behavior of the network.
        action_probs = self.policy_net(state_tensor)  # Get the action probabilities from the policy network

        # Sample an action from the categorical distribution defined by the action probabilities
        action = action_probs.sample()  # Sample an action index from the distribution

        return action.item(), action_probs.log_prob(action)  # Return the selected action index and its log probability
    

    def step(self, log_prob, reward, done):
        """Update the policy network based on the log probability of the action taken and the reward received
        
        Args:
            log_prob (torch.Tensor): The log probability of the action taken, used for calculating the policy gradient loss
            reward (float): The reward received after taking the action, used for calculating the discounted return for the policy gradient update
            done (bool): Whether the episode has ended after taking the action
        """
        # Store the log probability, reward and done mask for the current time step, which will be used for calculating the policy gradient loss at the end of the episode
        self.log_probs.append(log_prob)
        self.rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
        self.dones.append(torch.tensor(done, dtype=torch.int16, device=self.device))

    def update_policy(self):
        """Update the policy network at the end of an episode using the collected log probabilities and rewards"""

        # Calculate the discounted returns for each time step in the episode
        discounted_returns = self.compute_discounted_returns()  # [episode_length]

        # Normalize the discounted returns for better training stability
        discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-8) # [episode_length]

        # Convert the list of log probabilities to a tensor for calculating the policy gradient loss
        log_probs_tensor = torch.cat(self.log_probs)  # [episode_length]

        # Calculate the policy gradient loss as the negative of the sum of the log probabilities of the actions taken multiplied by their corresponding discounted returns
        # We take negative because we want to maximize the expected return, which is equivalent to minimizing the negative expected return.
        policy_loss = -1.0 * torch.mean(log_probs_tensor * discounted_returns)  # Scalar loss value

        # Perform a backward pass to calculate the gradients of the loss with respect to the policy network's parameters
        self.optimizer.zero_grad()  # Clear any existing gradients
        policy_loss.backward()  # Backpropagate the loss to compute gradients
        self.optimizer.step()  # Update the policy network's parameters using the computed gradients

        # Clear the log probabilities and rewards for the next episode
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()


    def compute_discounted_returns(self):
        """Compute the discounted returns for each time step in the episode, used for calculating the policy gradient update at the end of the episode. 
        
        Returns:
            torch.Tensor: A tensor containing the discounted returns for each time step in the episode, used for calculating the policy gradient update at the end of the episode. 
            The discounted return at each time step is calculated as the sum of the rewards from that time step until the end of the episode, discounted by the factor gamma.
            Shape: [episode_length]
        """
        discounted_returns = []
        R = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = reward + self.gamma * R * (1 - done)  # Only include future rewards if the episode has not ended at this time step
            discounted_returns.insert(0, R)  # Insert at the beginning to maintain correct order

        return torch.stack(discounted_returns)
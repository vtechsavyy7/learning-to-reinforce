# An RL agent that uses Deep Q-Networks (DQN) to learn how to play the Space Invaders game from the OpenAI Gym environment. 
# The agent preprocesses the frames from the environment, stacks them to create a state representation, 
# and uses a convolutional neural network to estimate Q-values for each action. 
# The agent also uses a replay buffer to store experience tuples and sample mini-batches for training the DQN.

from DQN import DQN
from preprocess import preprocess_frame, queue_new_frame
from replay_buffer import ReplayBuffer

import torch
from torch import optim
import numpy as np

class DQNAgent:
    def __init__(self, input_shape, num_actions, replay_buffer_capacity, device, learning_rate, gamma, batch_size, update_target_every, replay_start_size):
        """Initialize the DQN agent with the given parameters
        
        Args:
            input_shape (tuple): The shape of the input state representation (e.g., (4, 84, 84) for a stack of 4 grayscale frames)
            num_actions (int): The number of possible actions in the environment (e.g., 6 for Space Invaders)
            replay_buffer_capacity (int): The maximum number of experience tuples to store in the replay buffer
            device (torch.device): The device to store the tensors on
            learning_rate (float): The learning rate for the optimizer
            gamma (float): The discount factor for future rewards
            batch_size (int): The batch size for sampling from the replay buffer during training
            update_target_every (int): The number of steps to take before updating the target network with the weights of the main DQN
            replay_start_size (int): The minimum number of experience tuples required in the replay buffer before starting training the DQN
        """
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.replay_start_size = replay_start_size

        self.policy_net = DQN(input_shape, num_actions).to(device)
        self.target_net = DQN(input_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set the target network to always run in evaluation mode
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity, device)

        self.agent_step = 0  # Counter to keep track of the number of steps taken by the agent, used for updating the target network at regular intervals

    def select_action(self, state, epsilon):
        """Select an action using an epsilon-greedy policy based on the current state and the policy network's Q-value estimates
        
        Args:
            state (numpy array): The current state representation (e.g., a queue of 4 preprocessed frames)
            epsilon (float): The probability of selecting a random action (exploration) versus selecting the action with the highest Q-value estimate (exploitation)
        
        Returns:
            int: The index of the selected action
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.policy_net.num_actions)  # Explore: select a random action
        else:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float32, device=self.device)  # Add batch dimension

            # Move the policy network to evaluation mode to disable dropout and batch normalization during action selection
            self.policy_net.eval()
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            self.policy_net.train()  # Move the policy network back to training mode

            return q_values.argmax().item()  # Exploit: select the action with the highest Q-value estimate
        
    def step(self, state, action, reward, next_state, done):
        """Store the experience tuple in the replay buffer and update the DQN if enough experience has been collected
        
        Args:
            state (numpy array): The current state representation (e.g., a queue of 4 preprocessed frames)
            action (int): The action taken by the agent in the current state
            reward (float): The reward received after taking the action
            next_state (numpy array): The next state representation after taking the action
            done (bool): Whether the episode has ended after taking the action
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

        # Increment the step counter for the agent, which is used to determine when to update the target network
        # But also reset it based on update_target_every to avoid overflow and to keep track of the number of steps since the last target network update
        self.agent_step = (self.agent_step + 1) % self.update_target_every

        # Only update the DQN if we have collected enough experience tuples in the replay buffer to sample a full batch for training.
        #  This helps to ensure that the DQN is trained on a diverse set of experiences and prevents overfitting to a small number of samples early in training.
        if len(self.replay_buffer) >= self.replay_start_size:
            # Sample a batch of experience tuples from the replay buffer and use them to update the policy network (DQN)
            experiences = self.replay_buffer.sample(self.batch_size)
            self.update_policy_net(experiences)

        # Check if it's time to update the DQN and the target network based on the number of steps taken by the agent and the amount of experience collected in the replay buffer
        if self.agent_step == 0:
            # Update the target network at regular intervals. 
            # Direcly copy the weights from the policy network to the target network every 'update_target_every' steps to stabilize training.
            self.target_net.load_state_dict(self.policy_net.state_dict())


    def update_policy_net(self, experiences):
        """Update the policy network (DQN) using a batch of experience tuples sampled from the replay buffer
        
        Args:
            experiences (tuple): A tuple containing batches of states, actions, rewards, next_states, and done flags sampled from the replay buffer
        """
        states, actions, rewards, next_states, dones = experiences

        # Compute the current Q-value estimates for the actions taken in the sampled batch of experiences using the policy network
        # Gather the Q-values corresponding to the actions taken by the agent in the sampled batch of experiences, 
        # which will be used to compute the loss against the target Q-values.
        current_pred_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1) # [batch_size]

        # Compute the target Q-values for the next states using the target network. 
        # The target Q-value is calculated as the reward received plus the discounted maximum Q-value estimate for the next state, but only if the episode has not ended (done flag is False).
        with torch.no_grad():
            target_net_out = self.target_net(next_states)  # [batch_size, num_actions]

            # Note: torch.max outputs a tuple of (max_values, indices_of_max_values)
            max_next_q_values = target_net_out.max(dim=1)[0]  # [batch_size]
            target_q_values = rewards + (self.gamma * max_next_q_values * (~dones))  # [batch_size]

        # Compute the loss between the current Q-value estimates and the target Q-values using mean squared error loss
        loss = torch.nn.functional.mse_loss(current_pred_q_values, target_q_values)

        # Perform a gradient descent step to update the weights of the policy network based on the computed loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

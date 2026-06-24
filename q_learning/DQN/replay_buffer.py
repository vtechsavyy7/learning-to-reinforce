# a class to store a fixed-size buffer of experience tuples for training the DQN agent
import random
from typing import NamedTuple
import numpy as np
from collections import deque

import torch

class Experience(NamedTuple):
    state: object
    action: int
    reward: float
    next_state: object
    done: bool


class ReplayBuffer:
    def __init__(self, capacity, device):
        """Initialize the replay buffer with a given capacity (maximum number of experience tuples to store)
        
        Args:
            capacity (int): The maximum number of experience tuples to store in the buffer
            device (torch.device): The device to store the tensors on
        """

        self.buffer = deque(maxlen=capacity)
        self.device = device
        self.seed = random.seed(42)  # Set a fixed seed for reproducibility of random sampling from the buffer

    def push(self, state, action, reward, next_state, done):
        """Add a new experience named tuple (state, action, reward, next_state, done) to the buffer. 
        
        Args:
            state (numpy array): The current state representation (e.g., a stack of 4 preprocessed frames)
            action (int): The action taken by the agent in the current state
            reward (float): The reward received after taking the action
            next_state (numpy array): The next state representation after taking the action
            done (bool): Whether the episode has ended after taking the action
        """
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(list(self.buffer), batch_size)
        state, action, reward, next_state, done = zip(*batch)

        # Convert the lists to Torch tensors
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.int64, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
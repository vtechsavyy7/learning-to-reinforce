from collections import namedtuple

import numpy as np
from utils import State, get_init_q, get_init_v


# Named tuple to store the state, action and reward for each step in an episode
SAR = namedtuple('SAR', ['state', 'action', 'reward'])

# Named tuple to store the state and action
SA = namedtuple('SA', ['state', 'action'])

class Episode:
    """Class to represent an episode in the Monte Carlo algorithm. It stores the sequence of states, actions and rewards for each step in the episode,
      as well as the number of steps taken and a dictionary to keep track of the first occurrence of each state in the episode (used for first visit Monte Carlo)."""
    def __init__(self, start_state : State):
        """Initialize the episode.
        
        Args:
            start_state (State): the starting state of the episode
        """
        self.sars : list[SAR] = []  # list of (state, action, reward) tuples
        self.num_steps = 0
        self.state_action_to_first_occurence : dict[SA, int] = {}
        self.curr_state = start_state

    def add_step(self, state, action, reward): 
        """Add a step to the episode.

        Args:
            state: The state at the current step.
            action: The action taken at the current step.
            reward: The reward received after taking the action. 
        """
        self.sars.append(SAR(state, action, reward))

        if (state, action) not in self.state_action_to_first_occurence:
            self.state_action_to_first_occurence[(state, action)] = self.num_steps
        
        self.num_steps += 1

# Enum for the type of Monte Carlo algorithm to use: first visit or every visit
class MonteCarloType:
    FIRST_VISIT = 1
    EVERY_VISIT = 2

class MonteCarloSolver:
    """Class to run a Monte Carlo algorithm for policy evaluation. Objective is to estimate the value function for a given policy."""

    def __init__(self, env, mc_type: MonteCarloType, v0_val=0):
        """Initialize the Monte Carlo predictor.
        
        Args:
            env (Environment): the environment to run the algorithm on
            mc_type (MonteCarloType): the type of Monte Carlo algorithm to use
            v0_val (float): the initial value for all states in the value function
        """
        self.env = env
        self.mc_type = mc_type

        self.action_value_func = get_init_q(env.n, len(env.actions), v0_val, env.e_x, env.e_y)  # initialize the action-value function with the given initial value for all state-action pairs
        self.returns : dict[SA, list[float]] = {}   # dictionary to store the returns for each state-action pair, used to calculate the average return for each state-action pair and update the action-value function accordingly

        # Initialize an empty policy:
        self.policy = None

    def solve(self, policy, num_episodes, max_steps_per_episode, gamma, epsilon):
        """Evaluate the policy by running the Monte Carlo algorithm for a given number of episodes.

        Args:
            policy (numpy array): the policy to evaluate.
            num_episodes (int): the number of episodes to run the algorithm for.
            max_steps_per_episode (int): the maximum number of steps to take in each episode before terminating it.
            gamma (float): the discount factor to use when calculating the returns.
            epsilon (float): the probability of selecting a random action in the epsilon-soft policy improvement step.
        """

        # Initialize the policy to evaluate
        self.policy = policy

        for episode in range(num_episodes):

            # Randomly select a starting state that is not terminal
            while True:
                start_x = np.random.randint(0, self.env.n)
                start_y = np.random.randint(0, self.env.n)
                if not self.env.is_terminal_state(State(start_x, start_y)):
                    break
            
            # Create a new episode starting from the selected state
            episode = Episode(State(start_x, start_y))

            while episode.num_steps < max_steps_per_episode:

                # Select an action according to the current policy
                action_probs = self.policy[episode.curr_state.x, episode.curr_state.y]
                action = np.random.choice(len(action_probs), p=action_probs)

                # Take the action and observe the next state and reward
                next_state, reward = self.env.step(episode.curr_state, action)

                # Add the step to the episode
                episode.add_step(episode.curr_state, action, reward)

                # If the episode has ended, break the loop
                if self.env.is_terminal_state(next_state):
                    break

                # Update the current state
                episode.curr_state = next_state

            # End of current episode: Now update the value function using the returns from this episode
            self.update_action_value_function_and_policy(episode, gamma, epsilon)

    def update_action_value_function_and_policy(self, episode, gamma, epsilon):
        """Update the action-value function and the policy using the returns from the given episode.

        Args:
            episode (Episode): the episode to use for updating the action-value function.
            gamma (float): the discount factor to use when calculating the returns.
            epsilon (float): the probability of selecting a random action in the epsilon-soft policy improvement step.
        """
        # Initialize the return G to 0
        G = 0

        for step in range(episode.num_steps - 1, 0, -1):

            # Update the return G by adding the reward from the current step and discounting the previous return
            G = gamma * G + episode.sars[step].reward

            # Get the state and action from the current step
            state = episode.sars[step].state
            action = episode.sars[step].action

            # For first visit Monte Carlo: only update the value function for the first occurrence of each state in the episode
            # For every visit Monte Carlo: update the value function for every occurrence of each state in the episode
            if episode.state_action_to_first_occurence[(state, action)] >= step or self.mc_type == MonteCarloType.EVERY_VISIT:
                # Add the return G to the list of returns for the state
                if (state, action) not in self.returns:
                    self.returns[(state, action)] = []
                self.returns[(state, action)].append(G)

                # Update the value function for the state by calculating the average return for that state
                self.action_value_func[state.x, state.y, action] = sum(self.returns[(state, action)]) / len(self.returns[(state, action)])

                # Update the policy for the state by selecting the action with the highest value
                # This essentially makes the policy greedy with respect to the current action-value function, 
                # which is a common approach in Monte Carlo control algorithms to ensure convergence to the optimal policy.
                best_action = np.argmax(self.action_value_func[state.x, state.y], axis=-1)

                # Update the policy using an epsilon-soft policy improvement step: 
                # We make the best action the most likely to be selected, 
                # but we also assign a small probability to the other actions to ensure exploration.
                # That small probability is given by epsilon, which is divided equally among the non-best actions.
                num_actions = len(self.env.actions)
                for a in range(num_actions):
                    if a == best_action:
                        self.policy[state.x, state.y, a] = 1 - epsilon + (epsilon / num_actions)
                    else:
                        self.policy[state.x, state.y, a] = epsilon / num_actions

    def get_value_function(self):
        """Get the value function for the current policy by taking the maximum action-value for each state.

        Returns:
            value_func (numpy array): the value function for the current policy.
        """
        value_func = np.max(self.action_value_func, axis=2)
        return value_func
    
    def get_policy(self):
        """Get the policy being solved for.

        Returns:
            policy (numpy array): the policy being solved for.
        """
        return self.policy
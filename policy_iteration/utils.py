
import numpy as np
import random

import matplotlib.pyplot as plt
from matplotlib import colors

class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

# Define the environment class
class Environment:
    def __init__(self, n, actions, rewards, e_x, e_y):
        self.n = n
        self.actions = actions   # list of possible actions
        self.rewards = rewards   # 2D numpy array representing the rewards
        self.e_x = e_x   # x value of the terminal state
        self.e_y = e_y   # y value of the terminal state

    def is_terminal_state(self, state: State):
        """Check if the given state is terminal."""
        return state.x == self.e_x and state.y == self.e_y
    
    def step(self, state: State, action: int):
        """Take a step in the environment given a state and an action.

        Args:
            state (State): current state
        """
        next_state = get_next_state(state.x, state.y, action, self.n)

        reward = self.rewards[next_state[0], next_state[1]]

        return State(next_state[0], next_state[1]), reward

def build_grid(n, p_barrier, r_barrier, seed_nr):
    """Build an NxN grid with start and end cells, as well as some barrier cells.

    Args:
        n (int): length and width of the grid
        p_barrier (float): probability of a cell being a barrier
        r_barrier (int): reward for the barrier cells

    Returns:
        env (Environment): grid world environment
    """

    # Define set of possible actions: go left (0), up (1), right (2) or down (3)
    actions = [0, 1, 2, 3]

    # Define start and end cells -> these will have value 0
    random.seed(seed_nr)
    e_x = random.randrange(n)
    e_y = random.randrange(n)

    # Define barrier cells -> these will have barrier reward. All other have -1 reward
    rewards = (-1) * np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != e_x or j != e_y:
                p = random.uniform(0, 1)
                if p < p_barrier:
                    rewards[i, j] = r_barrier

    # Create environment
    env = Environment(n, actions, rewards, e_x, e_y)

    return env

def get_init_v(n, v0, e_x, e_y):
    """Defines initial value function v_0

    Args:
        n (int): length and width of the grid
        v0 (float): initial value for the value function (equal for every state)
        e_x (int): x value of the end cell
        e_y (int): y value of the end cell

    Returns:
        v0 (array): initial value function
    """

    v0 = v0 * np.ones((n, n))

    # Value function of terminal state must be 0
    v0[e_x, e_y] = 0

    return v0


def get_init_q(n, num_actions, q0_val, e_x, e_y):
    """Defines initial action-value function q_0

    Args:
        n (int): length and width of the grid
        num_actions (int): number of possible actions in each state
        q0_val (float): initial value for the action-value function (equal for every state-action pair)

    Returns:
        q0 (array): initial action-value function
    """

    q0 = q0_val * np.ones((n, n, num_actions))

    # Value function of terminal state must be 0 for every action
    q0[e_x, e_y, :] = 0.0

    return q0


def get_equiprobable_policy(n):
    """Defines the equiprobable policy. Policy is a matrix s.t.
        pi[x, y, a] = Pr[A = a | S = (x,y)]

    Actions are:
        * 0: go left
        * 1: go up
        * 2: go right
        * 3: go down

    Args:
        n (int): length and width of the grid

    Returns:
        pi (array): numpy array representing the equiprobably policy
    """

    pi = 1/4 * np.ones((n, n, 4))
    return pi


# Get a policy that is skewed towards going right and down
def get_skewed_policy(n):
    """Defines a policy that is skewed towards going right and down. Policy is a matrix s.t.
        pi[x, y, a] = Pr[A = a | S = (x,y)]

    Actions are:
        * 0: go left
        * 1: go up
        * 2: go right
        * 3: go down

    Args:
        n (int): length and width of the grid

    Returns:
        pi (array): numpy array representing the skewed policy
    """

    pi = np.zeros((n, n, 4))
    pi[:, :, 0] = 0.1   # left
    pi[:, :, 1] = 0.1   # up
    pi[:, :, 2] = 0.4   # right
    pi[:, :, 3] = 0.4   # down

    return pi


def get_next_state(x, y, a, n):
    """Computes next state from current state and action.

    Args:
        x (int): x value of the current state
        y (int): y value of the current state
        a (int): action
        n (int): length and width of the grid

    Returns:
        s_prime_x (int): x value of the next state
        s_prime_y (int): y value of the next state
    """

    # Compute next state according to the action
    if a == 0:
        s_prime_x = x
        s_prime_y = max(0, y - 1)
    elif a == 1:
        s_prime_x = max(0, x - 1)
        s_prime_y = y
    elif a == 2:
        s_prime_x = x
        s_prime_y = min(n - 1, y + 1)
    else:
        s_prime_x = min(n - 1, x + 1)
        s_prime_y = y

    return s_prime_x, s_prime_y


def bellman_update(env, v, old_v, x, y, pi, gamma):
    """Applies the Bellman update rule to the value function

    Args:
        env (Environment): grid world environment
        v (array): numpy array representing the value function
        old_v (array): numpy array representing the value function on the last iteration
        x (int): x value position of the current state
        y (int): y value position of the current state
        pi (array): numpy array representing the policy
        gamma (float): gamma parameter (between 0 and 1)
    """

    # The value function on the terminal state always has value 0
    if x == env.e_x and y == env.e_y:
        return None

    total = 0

    for a in env.actions:
        # Get next state
        s_prime_x, s_prime_y = get_next_state(x, y, a, env.n)

        total += pi[x, y, a] * (env.rewards[s_prime_x, s_prime_y] + gamma * old_v[s_prime_x, s_prime_y])

    # Update the value function
    v[x, y] = total


def define_new_policy(pi, x, y, best_actions, actions):
    """Defines a new policy given the new best actions.

    Args:
        pi (array): numpy array representing the policy
        x (int): x value position of the current state
        y (int): y value position of the current state
        best_actions (list): list with best actions
        actions (list): list of every possible action
    """

    prob = 1/len(best_actions)

    for a in actions:
        pi[x, y, a] = prob if a in best_actions else 0


from monte_carlo_utils import SAR, Episode, MonteCarloType, MonteCarloPredictor
from utils import *

class OffPolicyMonteCarloPredictor(MonteCarloPredictor):
    """Class to run an off-policy Monte Carlo algorithm for policy evaluation. Objective is to estimate the value function for a given target policy, while following a different behavior policy to generate the episodes."""

    def __init__(self, env, mc_type: MonteCarloType, v0_val=0):
        """Initialize the off-policy Monte Carlo predictor.
        
        Args:
            env (Environment): the environment to run the algorithm on
            mc_type (MonteCarloType): the type of Monte Carlo algorithm to use
            v0_val (float): the initial value for all states in the value function
        """
        super().__init__(env, mc_type, v0_val)

    
    def evaluate_off_policy(self, target_policy, behavior_policy, num_episodes):
        """Evaluate the target policy using the off-policy Monte Carlo algorithm.

        Args:
            target_policy: the policy to evaluate
            behavior_policy: the policy to follow to generate the episodes
            num_episodes: the number of episodes to run the algorithm for
        """
        for episode_num in range(num_episodes):
            # Generate an episode following the behavior policy
            start_state = State(0, 0)  # start state is always (0, 0)
            episode = Episode(start_state)

            while not self.env.is_terminal_state(episode.curr_state):
                action = behavior_policy[episode.curr_state.x, episode.curr_state.y]
                next_state, reward = self.env.step(episode.curr_state, action)
                episode.add_step(next_state, action, reward)
                episode.curr_state = next_state
            
            # Calculate the return G for each step in the episode and update the action-value function accordingly
            G = 0
            for step in reversed(range(episode.num_steps)):
                state = episode.sars[step].state
                action = episode.sars[step].action
                reward = episode.sars[step].reward

                G += reward
from copy import deepcopy

import numpy as np

from .agent import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(self, learning_rate: int=0.1, discount_factor: int=0.95, epsilon: int=0.1) -> None:
        """Initialize the Q-Learning Agent."""
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        self.last_state = None
        self.last_action = None

    

    def select_action(self, observation: dict, mask: list) -> list:
        """Select action using epsilon-greedy policy."""
        # Convert observation to discrete state
        state = self._discretize_state(observation)

        # Initialize state in Q-table if not seen before
        if state not in self.q_table:
            self.q_table[state] = np.zeros(625)

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            mask_indices = [self._action_to_index(valid_action) for valid_action in mask]
            random_action_index = np.random.choice(mask_indices)
            action = self._index_to_action(random_action_index)
            
        else:
            q_values = deepcopy(self.q_table[state])
            mask_indices = [self._action_to_index(valid_action) for valid_action in mask]
            
            invalid_actions = [i for i in range(625) if i not in mask_indices]
            q_values[invalid_actions] = -np.inf
            action = self._index_to_action(np.argmax(q_values))
            

        self.last_state = state
        self.last_action = action

        return action

    def update(self, reward: float, next_observation: dict=None) -> None:
        """Update Q-values using the Q-learning update rule."""
        if self.last_state is None or self.last_action is None:
            return

        if next_observation is None:
            next_max_q = 0
        else:
            next_state = self._discretize_state(next_observation)
            if next_state not in self.q_table:
                self.q_table[next_state] = np.zeros(625)
            next_max_q = np.max(self.q_table[next_state])

        # Q-learning update
        current_q = self.q_table[self.last_state][self.last_action]
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[self.last_state][self.last_action] = new_q

    def save_policy(self, filename):
        """Save the Q-table to a file."""
        np.save(filename, self.q_table)

    def load_policy(self, filename):
        """Load the Q-table from a file."""
        self.q_table = np.load(filename, allow_pickle=True).item()

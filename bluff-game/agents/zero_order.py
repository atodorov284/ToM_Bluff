from copy import deepcopy

import numpy as np

from .agent import BaseAgent


class ZeroOrderAgent(BaseAgent):
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
    ) -> None:
        """Initialize the Simplified Q-Learning Agent."""
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        self.last_state = None
        self.last_action = None
        self.ACTION_CHALLENGE = [0, 0, 0, 0]
        self.NUM_ACTIONS = 9  # challenge + 4 truth + 4 bluff

    def select_action(self, observation: dict, mask: list) -> list:
        """Select action using epsilon-greedy policy."""
        self.current_rank = observation["current_rank"]
        hand_freq = observation["hand"]

        # Get current state
        state = self._get_state(observation)
        valid_actions = self._get_valid_actions(mask, hand_freq)

        if not valid_actions:
            return mask[0]  # Fallback to any valid action

        # Initialize state in Q-table if not seen before
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.NUM_ACTIONS)

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action = np.random.choice(valid_actions)
        else:
            q_values = deepcopy(self.q_table[state])
            invalid_actions = [
                i for i in range(self.NUM_ACTIONS) if i not in valid_actions
            ]
            q_values[invalid_actions] = -np.inf
            action = np.argmax(q_values)

        self.last_state = state
        self.last_action = action

        full_action = self._convert_action_to_full(action, hand_freq)
        full_action = list(map(int, full_action))

        return full_action

    def update(self, reward: float, next_observation: dict = None) -> None:
        """Update Q-values using the Q-learning update rule."""
        if self.last_state is None or self.last_action is None:
            return

        if next_observation is None:
            next_max_q = 0
        else:
            next_state = self._get_state(next_observation)
            if next_state not in self.q_table:
                self.q_table[next_state] = np.zeros(self.NUM_ACTIONS)
            next_max_q = np.max(self.q_table[next_state])

        # Q-learning update
        current_q = self.q_table[self.last_state][self.last_action]
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[self.last_state][self.last_action] = new_q

from collections import defaultdict
from copy import deepcopy

import numpy as np

from agents.agent import BaseAgent


class FirstOrderAgent(BaseAgent):
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
    ) -> None:
        """Initialize the first-order ToM agent."""
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        self.last_state = None
        self.last_action = None
        self.ACTION_CHALLENGE = [0, 0, 0, 0]
        self.NUM_ACTIONS = 9  # challenge + 4 truth + 4 bluff

    def estimate_opponent_action(self, observation: dict) -> int:
        """Estimate the opponent's most likely action based on observation."""
        opponent_states = []
        for state, actions in self.q_table.items():
            if state[3] == observation["cards_other_agent_played"]:
                opponent_states.append((state, actions))
        if not opponent_states:
            return None
        state_action_counts = defaultdict(int)
        for state, actions in opponent_states:
            most_likely_action = np.argmax(actions)
            state_action_counts[most_likely_action] += 1
        return max(state_action_counts, key=state_action_counts.get)

    def select_action(self, observation: dict, mask: list) -> list:
        """
        FOR DECIDING WHETHER TO CHALLENGE OR NOT
                If you believe opponent is bluffing, then increase the q value for challenging them.
                if they dont play the number of cards you believe, then fall back to using the q table to figure out if you should challenge them.
                Else figure out whether opponent will challenge you and play bluff or truthfully accordingly.

        FOR DECIDING WHETHER TO BLUFF OR PLAY TRUTHFULLY
                Do it iteratively. Simulate possible actions you can make and using the model estimate what the opponent would do. An action is selected if it is valid.
                Valid actions in this case are for example if you simulate playing bluff, and you think your opponent would not challenge you, or if you played truthfully
                and you think your opponent would challenge you. Essentially whenever you would trick your opponent you have a valid action and the action you actually play is
                the valid action that you estimate to bring you the most reward.

        """

        self.current_rank = observation["current_rank"]
        hand_freq = observation["hand"]

        # Get current state
        state = self._get_state(observation)
        valid_actions = self._get_valid_actions(mask, hand_freq)

        if not valid_actions:
            return mask[0]

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.NUM_ACTIONS)

        # Estimate opponent's most likely action
        opponent_action = self.estimate_opponent_action(observation)
        if opponent_action == 0 and self.ACTION_CHALLENGE in valid_actions:
            return self.ACTION_CHALLENGE

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
        return self._convert_action_to_full(action, hand_freq)

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

        current_q = self.q_table[self.last_state][self.last_action]
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[self.last_state][self.last_action] = new_q

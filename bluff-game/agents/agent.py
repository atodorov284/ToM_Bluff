from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, action_space: np.ndarray, mask: np.ndarray) -> int:
        """
        Abstract method to select an action based on the action space and the mask.
        action_space: A probability distribution or set of probabilities for each action.
        mask: A binary mask of valid actions.
        """
        pass

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        """
        Update the agent's knowledge based on the action taken and the reward received.
        Args:
            action: The arm that was pulled.
            reward: The reward received after pulling the arm.
        """
        pass

    def _discretize_state(self, observation: dict) -> tuple:
        """Convert the observation dictionary into a hashable state tuple."""
        # Extract the hand frequency vector
        hand_freq = tuple(observation["hand"])  # Already in the format we want

        # Discretize pile size into 3 categories
        pile_size = observation["central_pile_size"]
        if pile_size <= 4:
            discrete_pile = 0  # small
        elif pile_size <= 8:
            discrete_pile = 1  # medium
        else:
            discrete_pile = 2  # large

        # Get the current rank (already discrete 0-3)
        current_rank = observation["current_rank"]

        cards_last_played = observation["cards_other_agent_played"]

        # Return state tuple
        return (*hand_freq, discrete_pile, current_rank, cards_last_played)

    def _action_to_index(self, action: list) -> int:
        """Convert action to index in Q-table."""
        return action[0] * 125 + action[1] * 25 + action[2] * 5 + action[3]

    def _index_to_action(self, index: int) -> list:
        """Convert index in Q-table to action."""
        return [index // 125, (index % 125) // 25, (index % 25) // 5, index % 5]

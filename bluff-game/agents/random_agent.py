import numpy as np

from agents.agent import BaseAgent


class RandomAgent(BaseAgent):
    def select_action(self, observation: dict, mask: list) -> list:
        """Randomly select a valid action."""
        mask_indices = [self._action_to_index(valid_action) for valid_action in mask]
        random_action_index = np.random.choice(mask_indices)
        action = self._index_to_action(random_action_index)
        return action

    def update(self, reward: float, next_observation: dict=None) -> None:
        """Random agent doesn't learn, so this is a no-op."""
        pass

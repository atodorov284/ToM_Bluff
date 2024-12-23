import numpy as np

from agents.agent import BaseAgent


class DummyAgent(BaseAgent):
    def select_action(self, observation: dict, mask: list) -> list:
        """Randomly select a valid action."""
        mask_indices = [self._action_to_index(valid_action) for valid_action in mask]
        first_action = mask_indices[0]
        action = self._index_to_action(first_action)
        return action

    def update(self, reward: float, next_observation: dict = None) -> None:
        """Random agent doesn't learn, so this is a no-op."""
        pass

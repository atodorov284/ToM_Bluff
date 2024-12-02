import numpy as np

from agents.agent import BaseAgent


class RandomAgent(BaseAgent):
    def select_action(self, observation, mask):
        """Randomly select a valid action."""
        # Ensure mask is a numpy array and at least 1d
        mask = np.atleast_1d(mask)
        valid_actions = np.where(mask == 1)[0]
        return np.random.choice(valid_actions)

    def update(self, reward, next_observation=None):
        """Random agent doesn't learn, so this is a no-op."""
        pass

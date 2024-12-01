from agents.agent import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
    def select_action(self, observation, mask):
        """Randomly select a valid action."""
        # Ensure mask is a numpy array and at least 1d
        mask = np.atleast_1d(mask)
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) == 0:
            valid_actions = np.arange(5)  # If no valid actions, allow all
        return np.random.choice(valid_actions)

    def update(self, reward, next_observation=None):
        """Random agent doesn't learn, so this is a no-op."""
        pass
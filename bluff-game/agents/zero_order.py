import numpy as np

from .agent import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        self.last_state = None
        self.last_action = None

    def _discretize_state(self, observation):
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

        # Return state tuple
        return (*hand_freq, discrete_pile, current_rank)

    def select_action(self, observation, mask):
        """Select action using epsilon-greedy policy."""
        # Convert observation to discrete state
        state = self._discretize_state(observation)

        # Initialize state in Q-table if not seen before
        if state not in self.q_table:
            self.q_table[state] = np.zeros(5)  # 5 possible actions

        # Ensure mask is a numpy array and at least 1d
        mask = np.atleast_1d(mask)

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            valid_actions = np.where(mask == 1)[0]
            if len(valid_actions) == 0:
                valid_actions = np.arange(5)
            action = np.random.choice(valid_actions)
        else:
            q_values = self.q_table[state].copy()
            q_values[mask == 0] = -np.inf
            action = np.argmax(q_values)

        # Store state for delayed update
        self.last_state = state
        self.last_action = action

        return action

    def update(self, reward, next_observation=None):
        """Update Q-values using the Q-learning update rule."""
        if self.last_state is None or self.last_action is None:
            return

        if next_observation is None:
            # Terminal state
            next_max_q = 0
        else:
            next_state = self._discretize_state(next_observation)
            if next_state not in self.q_table:
                self.q_table[next_state] = np.zeros(5)
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

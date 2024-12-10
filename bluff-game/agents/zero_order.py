from copy import deepcopy
import numpy as np
from .agent import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self, learning_rate: float=0.1, discount_factor: float=0.95, epsilon: float=0.1) -> None:
        """Initialize the Simplified Q-Learning Agent."""
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        self.last_state = None
        self.last_action = None
        self.ACTION_CHALLENGE = [0, 0, 0, 0]
        self.NUM_ACTIONS = 9  # challenge + 4 truth + 4 bluff
        
    def _get_state(self, observation: dict) -> tuple:
        """Convert observation into state tuple (cards_of_curr_rank, other_cards, current_rank)."""
        hand_freq = observation["hand"]
        current_rank = observation["current_rank"]
        
        cards_of_curr_rank = hand_freq[current_rank]
        other_cards = sum(hand_freq) - cards_of_curr_rank

        num_cards_other_agent_played = observation["cards_other_agent_played"]

        pile_size = observation["central_pile_size"]

        # discretize pile size
        if pile_size < 3:
            discrete_pile = 0
        elif pile_size > 3 and pile_size < 7:
            discrete_pile = 1
        else:
            discrete_pile = 2
        
        return (cards_of_curr_rank, other_cards, current_rank, num_cards_other_agent_played, discrete_pile)

    def _convert_action_to_full(self, action: int, hand_freq: list) -> list:
        """Convert action number to full action format.
        0: Challenge
        1-4: Play 1-4 truthful cards
        5-8: Play 1-4 bluff cards
        """
        if action == 0:  # Challenge
            return self.ACTION_CHALLENGE
            
        full_action = [0, 0, 0, 0]
        if action <= 4:  # Truthful play
            num_cards = action
            full_action[self.current_rank] = num_cards
        else:  # Bluff
            num_cards = action - 4
            
            # Get cards from other ranks
            other_ranks = [i for i in range(4) if i != self.current_rank]
            available_cards = [(rank, hand_freq[rank]) for rank in other_ranks if hand_freq[rank] > 0]
            if not available_cards:
                return self.ACTION_CHALLENGE  # Fallback if we can't bluff
                
            # Sort ranks by priority:
            # 1. Ranks that have already passed (lower than current)
            # 2. Remaining ranks
            available_cards.sort(key=lambda x: (
                0 if x[0] < self.current_rank else 1,  # Prioritize past ranks
                -x[1]  # Then by number of cards (most cards first)
            ))
            
            # Use cards from multiple ranks if needed
            cards_needed = num_cards
            for rank, count in available_cards:
                if cards_needed <= 0:
                    break
                cards_to_use = min(count, cards_needed)
                full_action[rank] = cards_to_use
                cards_needed -= cards_to_use

        return full_action

    def _get_valid_actions(self, mask: list, hand_freq: list) -> list:
        """Get list of valid actions in our simplified action space."""
        valid_actions = []
        
        # Check if challenge is valid
        if any(np.array_equal(action, self.ACTION_CHALLENGE) for action in mask):
            valid_actions.append(0)

        # For each action in mask, determine if it corresponds to a valid play in our action space
        for mask_action in mask:
            if np.array_equal(mask_action, self.ACTION_CHALLENGE):
                continue
                
            max_cards = sum(mask_action)  # Maximum cards allowed by this mask action
            
            # Check if this is a truth play (only uses current rank)
            if mask_action[self.current_rank] > 0:  # If we can play any cards of current rank
                max_truth_cards = mask_action[self.current_rank]  # How many we can play truthfully
                for num_cards in range(1, max_truth_cards + 1):
                    if hand_freq[self.current_rank] >= num_cards:
                        valid_actions.append(num_cards)  # Actions 1-4
                    
            # Check if this could be a bluff play (uses other ranks)
            if mask_action[self.current_rank] == 0:
                # Calculate available cards in the required ranks
                available_cards = sum(min(mask_action[i], hand_freq[i]) 
                                   for i in range(4) if i != self.current_rank)
                # Can play any number of cards up to min(max_cards, available_cards)
                for num_cards in range(1, min(max_cards, available_cards) + 1):
                    valid_actions.append(num_cards + 4)  # Actions 5-8
                    
        return sorted(list(set(valid_actions)))
    
    
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
            invalid_actions = [i for i in range(self.NUM_ACTIONS) if i not in valid_actions]
            q_values[invalid_actions] = -np.inf
            action = np.argmax(q_values)

        self.last_state = state
        self.last_action = action
        
        return self._convert_action_to_full(action, hand_freq)

    def update(self, reward: float, next_observation: dict=None) -> None:
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

    def save_policy(self, filename):
        """Save the Q-table to a file."""
        np.save(filename, self.q_table)

    def load_policy(self, filename):
        """Load the Q-table from a file."""
        self.q_table = np.load(filename, allow_pickle=True).item()
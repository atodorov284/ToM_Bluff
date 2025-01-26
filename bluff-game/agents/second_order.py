import numpy as np
from typing import Dict, List, Tuple
from agents.zero_order import ZeroOrderAgent
from copy import deepcopy
from agents.agent import BaseAgent
import math

from agents.first_order_dev import FirstOrderAgent

class SecondOrderAgent(BaseAgent):
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.97, epsilon: float = 0.1):
        """Initialize FirstOrderAgent with a model of opponent's behavior."""
        # Create internal model of opponent as a zero-order agent
        self.opponent_model = FirstOrderAgent(learning_rate, discount_factor, epsilon)
        
        # Initialize own learning parameters
        self.lr = learning_rate
        self.epsilon = epsilon
        self.gamma = discount_factor
        self.q_table = {}  # State-action values for self
        
        # Track game state
        self.last_state = None
        self.last_action = None
        self.ACTION_CHALLENGE = [0, 0, 0, 0]
        self.NUM_ACTIONS = 9  # challenge + 4 truth + 4 bluff
        
        
    def custom_round(self, number: int):
        threshold = 0.8
        return math.floor(number) if number < (math.floor(number) + threshold) else math.ceil(number)
        
    def estimate_opponent_cards(self, observation: Dict) -> List[List[float]]:
        """
        Estimate probability distribution of opponent's cards based on observed game state.
        Returns list of probabilities for each rank.
        """
        total_cards = observation['cards_in_other_agent_hand']
        my_hand = observation['hand']
        
        # Initialize probabilities for each rank
        # We know total cards but not distribution
        # Use simple uniform distribution as baseline
        cards = []
        remaining_cards = [4 - my_hand[i] for i in range(4)]  # 4 cards per rank initially
        total_remaining = sum(remaining_cards)
        
        for i in range(4):
            if total_remaining > 0:
                prob = self.custom_round((remaining_cards[i] / total_remaining) * total_cards)
            else:
                prob = 0
            cards.append(prob)
            
        return cards
    
    def estimate_opponent_observation(self, last_action: int, observation: Dict, opponent_cards: List[float]) -> Dict:
        
        num_my_cards = sum(observation['hand'])
        
        opponent_observation = {
            "current_rank": int(observation['current_rank']),
            "central_pile_size": int(observation['central_pile_size']),
            "hand": np.array(opponent_cards).tolist(), 	# zero order hand estimated
            "cards_other_agent_played": last_action,   # number of cards we just played. REMEMBER TO UPDATE THIS IN SELECT ACTION!
            "cards_in_other_agent_hand": num_my_cards   # number of cards we are holding
        }
        
        return opponent_observation
        
    def estimate_opponent_bluff_probability(self, 
                                          observation: Dict, 
                                          opponent_cards: List[float]) -> float:
        """
        Estimate probability that opponent is bluffing based on:
        1. Estimated opponent cards
        2. Opponent's typical behavior (from Q-table)
        """
        current_rank = observation['current_rank']
        cards_opp_played = observation['cards_other_agent_played']
        
        if cards_opp_played == 0:  # No cards played to evaluate
            return 0.0
            
        # Probability opponent has enough cards of current rank
        prob_has_cards = (opponent_cards[current_rank] >= cards_opp_played)
        
        last_action = self.last_action
        
        opponent_observation = self.estimate_opponent_observation(last_action, observation, opponent_cards)
        
        # Use our q-table to for the opponent
        self.opponent_model.q_table = self.q_table
        
        # Use opponent model to estimate bluff probability
        opponent_state = self.opponent_model._get_state(opponent_observation)
        
        
        if opponent_state not in self.opponent_model.q_table:
            self.opponent_model.q_table[opponent_state] = np.zeros(self.NUM_ACTIONS)

        opponent_q_values= self.opponent_model.q_table[opponent_state]
        
        # Compare Q-values for truth vs bluff
        truth_value = max(opponent_q_values[1:5])  # Actions 1-4 are truth
        bluff_value = max(opponent_q_values[5:])   # Actions 5-8 are bluff
        
        
        # Combine evidence
        bluff_prob = (1 - prob_has_cards) * 0.8  # High weight on card probability
        
        if bluff_value >= truth_value:
            return bluff_prob
        
        return 0
        
        if bluff_value >= truth_value:
            bluff_prob += 0.2  # Small weight on Q-value comparison
            
        return min(bluff_prob, 1.0)
    
    def categorize_action_index(self, action_index: int) -> int:
        if action_index == 0:
            return 0 # challenge
        elif action_index >= 1 and action_index <= 4:
            return 1 # truth
        else:
            return 2 # bluff
    
    def wins_interaction(self, last_action: int, observation: Dict, opponent_cards: List[float], mask) -> bool:
        first_order_action = self.categorize_action_index(last_action)
        
        # Use our q-table to for the opponent
        self.opponent_model.q_table = self.q_table
                
        opponent_observation = self.estimate_opponent_observation(last_action, observation, opponent_cards)
        
        opponent_action = self.opponent_model.select_action(opponent_observation, mask)
        
        opponent_action = self._discretize_action(opponent_action)
        
        opponent_action = self.categorize_action_index(opponent_action)

        if first_order_action == 2 and opponent_action == 0:
            return False
        
        # In any other case, the interaction is successful
        return True
    
    def select_action(self, observation: Dict, mask: List) -> List[int]:
        """
        Select action based on:
        1. Whether to challenge (if opponent played)
        2. Whether to bluff or play truthfully (if playing cards)
        """
        #------------------------------------------------#
        # Interpretative ToM
        # First decision: Challenge or play
        opponent_cards = self.estimate_opponent_cards(observation)
        
        if observation['cards_other_agent_played'] > 0:
            bluff_prob = self.estimate_opponent_bluff_probability(observation, opponent_cards)
            
            # Challenge if high probability of bluff
            if bluff_prob >= 0.7 and self.ACTION_CHALLENGE in mask:
                self.last_action = 0
                return self.ACTION_CHALLENGE
            
            
        #------------------------------------------------#
        # Predictive ToM
        # Second decision: If playing cards, decide whether to bluff
        self.current_rank = observation['current_rank']
        hand = observation['hand']
        
        # Get current state
        state = self._get_state(observation)
        valid_actions = self._get_valid_actions(mask, hand)
        
        if not valid_actions:
            return mask[0]  # Fallback to any valid action
        
        # Initialize state in Q-table if not seen before
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.NUM_ACTIONS)
        
        predictive = False
        q_values = deepcopy(self.q_table[state])
        invalid_actions = [
            i for i in range(self.NUM_ACTIONS) if i not in valid_actions
        ]
        q_values[invalid_actions] = -np.inf
        
        sorted_q_values = sorted(q_values, reverse=True)
        
        # loop through sorted q value and see if you 'win' an interaction
        for value in sorted_q_values:
            action = q_values.tolist().index(value)
            if action in valid_actions:
                if self.wins_interaction(action, observation, opponent_cards, mask):
                    predictive = True
                    break
            
        #------------------------------------------------#
        # Zero Order ToM
        if not predictive:
            # If both interpretative and predictive ToM are unsuccessful, select action as zero-order
            if np.random.random() < self.epsilon:
                action = np.random.choice(valid_actions)
            else:
                action = np.argmax(q_values)
        
        #------------------------------------------------#
        # Convert Action to correct format and seva state and action
        
        self.last_state = state 
        self.last_action = action
        
        full_action = self._convert_action_to_full(action, hand)
        full_action = list(map(int, full_action))

        return full_action
    
    def update(self, reward: float, next_observation: Dict) -> None:
        """Update internal models based on observed reward."""
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

import numpy as np
from typing import Dict, List, Tuple
from agents.zero_order import ZeroOrderAgent

class FirstOrderAgent2:
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.97, epsilon: float = 0.1):
        """Initialize FirstOrderAgent with a model of opponent's behavior."""
        # Create internal model of opponent as a zero-order agent
        self.opponent_model = ZeroOrderAgent(learning_rate, discount_factor, epsilon)
        
        # Initialize own learning parameters
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q_table = {}  # State-action values for self
        
        # Track game state
        self.last_state = None
        self.last_action = None
        
    def estimate_opponent_cards(self, observation: Dict) -> List[List[float]]:
        """
        Estimate probability distribution of opponent's cards based on observed game state.
        Returns list of probabilities for each rank.
        """
        total_cards = observation['cards_in_other_agent_hand']
        my_hand = observation['hand']
        current_rank = observation['current_rank']
        
        # Initialize probabilities for each rank
        # We know total cards but not distribution
        # Use simple uniform distribution as baseline
        probs = []
        remaining_cards = [4 - my_hand[i] for i in range(4)]  # 4 cards per rank initially
        total_remaining = sum(remaining_cards)
        
        for i in range(4):
            if total_remaining > 0:
                prob = (remaining_cards[i] / total_remaining) * total_cards
            else:
                prob = 0
            probs.append(prob)
            
        return probs
        
    def estimate_opponent_bluff_probability(self, 
                                          observation: Dict, 
                                          opponent_cards: List[float]) -> float:
        """
        Estimate probability that opponent is bluffing based on:
        1. Estimated opponent cards
        2. Opponent's typical behavior (from Q-table)
        """
        current_rank = observation['current_rank']
        cards_played = observation['cards_other_agent_played']
        
        if cards_played == 0:  # No cards played to evaluate
            return 0.0
            
        # Probability opponent has enough cards of current rank
        prob_has_cards = (opponent_cards[current_rank] >= cards_played)
        
        # Get opponent's estimated Q-values for this state
        opponent_state = (
            opponent_cards[current_rank],  # cards of current rank
            sum(opponent_cards) - opponent_cards[current_rank],  # other cards
            current_rank,
            cards_played,
            observation['central_pile_size']
        )
        
        # Use opponent model to estimate bluff probability
        opponent_q_values = self.opponent_model.q_table.get(opponent_state, 
                                                          [0.0] * 9)  # 9 possible actions
        
        # Compare Q-values for truth vs bluff
        truth_value = max(opponent_q_values[1:5])  # Actions 1-4 are truth
        bluff_value = max(opponent_q_values[5:])   # Actions 5-8 are bluff
        
        # Combine evidence
        bluff_prob = (1 - prob_has_cards) * 0.8  # High weight on card probability
        if bluff_value > truth_value:
            bluff_prob += 0.2  # Small weight on Q-value comparison
            
        return min(bluff_prob, 1.0)
    
    def select_action(self, observation: Dict, valid_actions: List) -> List[int]:
        """
        Select action based on:
        1. Whether to challenge (if opponent played)
        2. Whether to bluff or play truthfully (if playing cards)
        """
        opponent_cards = self.estimate_opponent_cards(observation)
        
        # First decision: Challenge or play
        if observation['cards_other_agent_played'] > 0:
            bluff_prob = self.estimate_opponent_bluff_probability(observation, opponent_cards)
            
            # Challenge if high probability of bluff
            if bluff_prob > 0.7 and [0,0,0,0] in valid_actions:
                return [0,0,0,0]
        
        # Second decision: If playing cards, decide whether to bluff
        current_rank = observation['current_rank']
        my_hand = observation['hand']
        
        # Prefer truth if we have matching cards
        if my_hand[current_rank] > 0:
            # Find valid truthful actions
            for action in valid_actions:
                if sum(action) > 0 and action[current_rank] == sum(action):
                    return action
                    
        # If no good truthful plays, consider bluffing
        # Filter to valid non-challenge actions
        play_actions = [a for a in valid_actions if sum(a) > 0]
        if play_actions:
            return play_actions[0]  # Pick first valid play
            
        # Default to challenge if no other good options
        return [0,0,0,0]
    
    def update(self, reward: float, next_observation: Dict) -> None:
        """Update internal models based on observed reward."""
        # Update opponent model
        if self.last_state is not None:
            self.opponent_model.update(reward, next_observation)
            
        # Store current state for next update
        self.last_state = next_observation
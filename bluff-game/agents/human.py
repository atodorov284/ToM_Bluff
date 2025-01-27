from typing import List, Dict
from .agent import BaseAgent

class HumanAgent(BaseAgent):
    def __init__(self) -> None:
        """Initialize the Human Agent."""
        self.RANKS = ["ACE", "JACK", "QUEEN", "KING"]
        self.ACTION_CHALLENGE = [0, 0, 0, 0]

    def _print_game_state(self, observation: Dict) -> None:
        """Print the current game state in a human-readable format."""
        print("\n=== Current Game State ===")
        print(f"Current Rank: {self.RANKS[observation['current_rank']]}")
        print(f"Cards in central pile: {observation['central_pile_size']}")
        
        # Print hand in a readable format
        hand = observation['hand']
        print("\nYour hand:")
        for i, count in enumerate(hand):
            if count > 0:
                print(f"{self.RANKS[i]}: {count}")
        
        # Print information about other player
        if observation['cards_other_agent_played'] > 0:
            print(f"Last play: Opponent played {observation['cards_other_agent_played']} cards")

    def _get_action_input(self) -> List[int]:
        """Get action input from the human player."""
        print("\nEnter the number of cards you want to play for each rank.")
        print("Enter 'c' or 'challenge' to challenge the previous play.")
        print("Format: number of ACES JACKS QUEENS KINGS (e.g., '2 0 1 0' to play 2 ACES and 1 QUEEN)")
        
        while True:
            try:
                action_input = input("Your play: ").lower().strip()
                
                # Handle challenge input
                if action_input in ['c', 'challenge']:
                    return self.ACTION_CHALLENGE
                
                # Parse card counts
                counts = [int(x) for x in action_input.split()]
                if len(counts) != 4:
                    print("Please enter exactly 4 numbers (one for each rank)")
                    continue
                    
                if any(x < 0 for x in counts):
                    print("Cannot play negative number of cards")
                    continue
                    
                return counts
                
            except ValueError:
                print("Invalid input. Please enter numbers separated by spaces, or 'c' to challenge")

    def select_action(self, observation: Dict, mask: List[List[int]]) -> List[int]:
        """Get action from human player through terminal input."""
        self._print_game_state(observation)
        
        while True:
            action = self._get_action_input()
            
            # Convert action and mask to tuples for proper comparison
            action_tuple = tuple(action)
            mask_tuples = [tuple(m) for m in mask]
            
            # Check if action is valid (exists in mask)
            if action_tuple in mask_tuples:
                return action
                
            # Give specific feedback on why the action is invalid
            if action == self.ACTION_CHALLENGE and not any(tuple(m) == tuple(self.ACTION_CHALLENGE) for m in mask):
                print("Cannot challenge at this time")
            else:
                total_cards = sum(action)
                if total_cards == 0:
                    print("Must play at least one card (unless challenging)")
                elif total_cards > 4:
                    print("Cannot play more than 4 cards")
                else:
                    # Check if player has enough cards
                    hand = observation['hand']
                    for i, (want_play, have) in enumerate(zip(action, hand)):
                        if want_play > have:
                            print(f"You don't have {want_play} {self.RANKS[i]} (you only have {have})")
                            break
                    else:
                        print("This play is not valid for the current game state - remember you can only play 0-4 cards total")

    def update(self, reward: float, next_observation: Dict = None) -> None:
        """Display the reward to the human player."""
        pass
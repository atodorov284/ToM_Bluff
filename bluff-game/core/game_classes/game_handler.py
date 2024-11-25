from core.card_classes.deck import Deck
from core.card_classes.central_pile import CentralPile
from core.card_classes.hand import Hand
from core.card_classes.card import Card
from typing import List, Dict, Optional
    
class BluffGameMPD():
    def __init__(self, num_players: int = 2) -> None:
        """
        Initialize the Bluff game as an MDP.
        :param num_players: Number of players in the game.
        """
        self._num_players = num_players
        self._ranks = ["Ace", "Jack", "Queen", "King"]
        self._deck = Deck(ranks=self._ranks, num_copies=4)
        self._central_pile = CentralPile()
        self._hands = [Hand() for _ in range(self._num_players)]
        self._current_rank_index = 0
        self._current_turn = 0
        self._last_play = None
        self._excess_cards = []

        self._setup_game()
        
    def _setup_game(self) -> None:
        """
        Set up the game by shuffling and dealing cards equally among players.
        Excess cards are placed face-down in the central pile.
        """
        self._deck.shuffle()
        cards_per_player = len(self._deck.cards) // self._num_players
        for hand in self._hands:
            hand.add_cards(self._deck.draw(cards_per_player))
        self._central_pile.add_to_top(self._deck.cards)
        
    def get_state(self) -> Dict:
        """
        Get the current state of the game for RL agents.
        :return: A dictionary representing the game state.
        """
        return {
            "hands": [hand.count() for hand in self._hands],
            "central_pile_count": len(self._central_pile),
            "current_rank": self._ranks[self._current_rank_index],
            "last_play": self._last_play,
            "current_turn": self._current_turn,
        }
        
    def take_action(self, player: int, action: Dict) -> Optional[Dict]:
        """
        Perform an action for the current player.
        :param player: The index of the player taking the action.
        :param action: A dictionary describing the action.
        :return: Result of the action, including transitions and rewards.
        """
        if player != self._current_turn:
            raise ValueError("It's not this player's turn!")

        action_type = action["type"]
        if action_type == "play":
            return self._play_cards(player, action["cards"], action["claimed_rank"])
        elif action_type == "challenge":
            return self._challenge(player)
        else:
            raise ValueError("Invalid action type!")
        
        
    def _play_cards(self, player: int, cards: List[Card], claimed_rank: str) -> Dict:
        """
        Handle the play action where a player declares cards of a rank.
        :param player: Player index.
        :param cards: List of cards being played.
        :param claimed_rank: Declared rank of the cards.
        :return: Result of the action.
        """
        if claimed_rank != self._ranks[self._current_rank_index]:
            raise ValueError("Invalid rank declared!")

        self._hands[player].remove_cards(cards)
        self._central_pile.add_cards(cards)
        self._last_play = {"player": player, "cards": cards, "claimed_rank": claimed_rank}

        # Pass turn to the next player
        self._current_turn = (self._current_turn + 1) % self._num_players

        return {"success": True, "next_turn": self._current_turn}
    
    def _challenge(self, challenger: int) -> Dict:
        """
        Handle the challenge action.
        :param challenger: Player index of the challenger.
        :return: Result of the action.
        """
        if not self._last_play:
            raise ValueError("No play to challenge!")

        last_player = self._last_play["player"]
        claimed_rank = self._last_play["claimed_rank"]
        cards = self._last_play["cards"]

        all_match = all(card.rank == claimed_rank for card in cards)
        if all_match:
            self._hands[challenger].add_cards(self._central_pile.cards)
            self._central_pile.cards = []
        else:
            self._hands[last_player].add_cards(self._central_pile.cards)
            self._central_pile.cards = []

        self._last_play = None
        self._current_rank_index = (self._current_rank_index + 1) % len(self._ranks)
        self._current_turn = (last_player + 1) % self._num_players

        return {"challenge_successful": not all_match, "next_turn": self._current_turn}
        
    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        :return: True if a player has won, False otherwise.
        """
        for hand in self._hands:
            if hand.count() == 0:
                return True
        return False
    
    def get_winner(self) -> Optional[int]:
        """
        Determine the winner of the game.
        :return: Index of the winning player, or None if no winner yet.
        """
        for i, hand in enumerate(self._hands):
            if hand.count() == 0:
                return i
        return None
        
    
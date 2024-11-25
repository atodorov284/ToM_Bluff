from core.card_classes.card import Card
from typing import List
from copy import deepcopy
from core.card_classes.card_collection import CardCollection

class Deck(CardCollection):
    def __init__(self, ranks: List[str], num_copies: int = 4, include_jokers: bool = False) -> None:
        """
        Initialize a deck of cards.
        :param ranks: List of ranks (e.g., ['Ace', 'King', 'Queen', 'Jack']).
        :param num_copies: Number of copies of each rank (default: 4).
        :param include_jokers: Whether to include Joker cards.
        """
        super().__init__()
        self._cards = [Card(rank) for rank in ranks for _ in range(num_copies)]
        if include_jokers:
            self._cards.extend([Card("Joker") for _ in range(num_copies)])
        self.shuffle()

    def draw(self, num_cards: int = 1) -> List[Card]:
        """
        Draw a specified number of cards from the deck.
        :param num_cards: Number of cards to draw.
        :return: List of drawn cards.
        """
        if num_cards > len(self.cards):
            drawn_cards = deepcopy(self.cards)
            self.cards = []
        else: 
            drawn_cards = self.cards[:num_cards]
            self.cards = self.cards[num_cards:]
        return drawn_cards

    def __str__(self) -> str:
        """Detailed string representation of all the cards in the deck."""
        return ", ".join(str(card) for card in self.cards)



if __name__ == "__main__":
    ranks = ["Ace", "King", "Queen", "Jack"]

    deck = Deck(ranks, num_copies=4, include_jokers=False)

    deck.shuffle_cards()

    p1_hand = []
    p2_hand = []
    
    while deck:
        p1_hand += deck.draw(3)
        p2_hand += deck.draw(2)

    print(f"Player 1: {p1_hand}")
    print(f"Player 2: {p2_hand}")
    
    for card in zip(p1_hand, p2_hand):
        if card[0] == card[1]:
            print("Cards are the same")
        else:
            print("Cards are different")
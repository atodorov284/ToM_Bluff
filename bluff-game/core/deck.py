import random
from card import Card

class Deck:
    def __init__(self, ranks, num_copies=4, include_jokers=False):
        """
        Initialize a deck of cards.
        :param ranks: List of ranks (e.g., ['Ace', 'King', 'Queen', 'Jack']).
        :param num_copies: Number of copies of each rank (default: 4).
        :param include_jokers: Whether to include Joker cards.
        """
        self.cards = [Card(rank) for rank in ranks for _ in range(num_copies)]
        if include_jokers:
            self.cards.extend([Card("Joker") for _ in range(num_copies)])
        self.shuffle()

    def shuffle(self):
        """Shuffle the deck."""
        random.shuffle(self.cards)

    def draw(self, num_cards=1):
        """
        Draw a specified number of cards from the deck.
        :param num_cards: Number of cards to draw.
        :return: List of drawn cards.
        """
        if num_cards > len(self.cards):
            raise ValueError("Not enough cards in the deck!")
        drawn_cards = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return drawn_cards

    def __len__(self):
        """Return the number of cards remaining in the deck."""
        return len(self.cards)
    
    def __str__(self):
        """
        Detailed string representation of all the cards in the deck.
        Useful for debugging or visualizing the deck.
        """
        return ", ".join([str(card) for card in self.cards])


if __name__ == "__main__":
    ranks = ["Ace", "King", "Queen", "Jack"]

    deck = Deck(ranks, num_copies=4, include_jokers=False)

    deck.shuffle()

    p1_hand = []
    p2_hand = []
    
    while len(deck) > 0:
        p1_hand += deck.draw(2)
        p2_hand += deck.draw(2)

    print(f"Player 1: {p1_hand}")
    print(f"Player 2: {p2_hand}")
    
    for card in zip(p1_hand, p2_hand):
        if card[0] == card[1]:
            print("Cards are the same")
        else:
            print("Cards are different")
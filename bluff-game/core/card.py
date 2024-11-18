class Card:
    def __init__(self, rank, suit=None):
        """
        Initialize a card with a rank and an optional suit.
        In the Bluff game, suits are not used but may be useful for future extensions.
        """
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        """
        String representation of the card.
        For example: 'Ace', 'King', 'Jack', 'Queen' or 'Joker' (if suits are omitted).
        """
        return f"{self.rank}"

    def __eq__(self, other):
        """
        Equality comparison between two cards based on their rank (and suit if applicable).
        """
        return self.rank == other.rank and self.suit == other.suit

class Card:
    def __init__(self, rank: str) -> None:
        """
        Initialize a card with a rank and an optional suit.
        In the Bluff game, suits are not used but may be useful for future extensions.
        """
        self._rank = rank
        
    @property
    def rank(self) -> str:
        """
        Get the rank of the card.
        """
        return self._rank
        
    def __repr__(self) -> None:
        """
        String representation of the card.
        For example: 'Ace', 'King', 'Jack', 'Queen' or 'Joker' (if suits are omitted).
        """
        return f"{self.rank}"

    def __eq__(self, other: 'Card') -> bool:
        """
        Equality comparison between two cards based on their rank.
        """
        return self.rank == other.rank
    
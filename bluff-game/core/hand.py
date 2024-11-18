class Hand:
    def __init__(self):
        """Initialize an empty hand."""
        self.cards = []

    def add_cards(self, cards):
        """Add cards to the hand."""
        self.cards.extend(cards)

    def remove_cards(self, cards):
        """Remove specific cards from the hand."""
        for card in cards:
            self.cards.remove(card)

    def display(self):
        """Return a string representation of the cards in the hand."""
        return ", ".join(str(card) for card in self.cards)

    def count(self):
        """Return the number of cards in the hand."""
        return len(self.cards)

    def __repr__(self):
        return f"Hand({len(self.cards)} cards: {self.display()})"
from card_classes.card_collection import CardCollection

class Hand(CardCollection):
    def display(self) -> str:
        """Return a string representation of the cards in the hand."""
        return ", ".join(str(card) for card in self.cards)
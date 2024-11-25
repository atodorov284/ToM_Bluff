from typing import List
import random
from core.card_classes.card import Card


class CardCollection:
    def __init__(self) -> None:
        """Initialize an empty card collection."""
        self._cards = []

    @property
    def cards(self) -> List[Card]:
        """Get the list of cards."""
        return self._cards

    @cards.setter
    def cards(self, value: List[Card]) -> None:
        """Set the list of cards."""
        self._cards = value

    def add_cards(self, cards: List[Card]) -> None:
        """Add cards to the collection."""
        self._cards.extend(cards)

    def remove_cards(self, cards: List[Card]) -> None:
        """Remove specific cards from the collection."""
        for card in cards:
            self._cards.remove(card)

    def shuffle(self) -> None:
        """Shuffle the cards in the collection."""
        self._cards = random.sample(self._cards, len(self._cards))

    def __len__(self) -> int:
        """Return the number of cards in the collection."""
        return len(self._cards)

    def __repr__(self) -> str:
        """String representation of the collection."""
        return f"{self.__class__.__name__}({len(self._cards)} cards)"
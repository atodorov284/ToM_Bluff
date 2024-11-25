from core.card_classes.deck import Deck
from core.game_classes.game_handler import BluffGameMPD
from agents.agent import Agent


if __name__ == "__main__":
    game = BluffGameMPD(num_players=2)
    player_1 = Agent("Player 1")
    player_2 = Agent("Player 2")
    
    print(game.take_action(1, ))
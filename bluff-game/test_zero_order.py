import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agents.zero_order import QLearningAgent
from envs.bluff_env import env


def play_bluff_game(num_players: int = 2, episodes: int = 1, seed: int = 4242) -> None:
    """Play a game of Bluff with the specified number of players."""
    random.seed(seed)
    np.random.seed(seed)

    game_env = env(num_players=num_players)
    for episode in range(episodes):
        game_env.reset()
        obs, info = game_env.get_initial_observation()
        mask = np.atleast_1d(np.array(info["action_mask"]))

        # Needed as you do not need to update in the first step, need both players to actually play something
        prev_rewards = {"player_0": None, "player_1": None}

        while True:
            # Get current agent
            current_agent = game_env.agent_selection

            action = np.random.choice(mask)
            game_env.step(action)
            next_obs, reward, termination, truncation, info = game_env.last()

            mask = np.atleast_1d(np.array(info["action_mask"]))

            if termination or truncation:
                break

            obs = next_obs


if __name__ == "__main__":
    play_bluff_game()

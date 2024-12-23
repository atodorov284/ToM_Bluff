import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agents.random_agent import RandomAgent  # noqa: F401
from agents.zero_order import QLearningAgent
from envs.bluff_env import env
from utils import print_strategy_analysis


def play_bluff_game(num_players: int = 2, episodes: int = 8, seed: int = 1) -> None:
    """Play a game of Bluff with the specified number of players."""
    random.seed(seed)
    np.random.seed(seed)

    game_env = env(num_players=num_players, render_mode="human")

    agent_1 = QLearningAgent(learning_rate=0.1, discount_factor=1, epsilon=0.1)

    agent_0 = QLearningAgent(learning_rate=0.1, discount_factor=1, epsilon=0.1)

    wins_agent_0 = 0
    wins_agent_1 = 0

    for episode in range(episodes):
        should_swap = random.random() < 0.5

        agents = {
            "player_0": agent_1 if should_swap else agent_0,
            "player_1": agent_0 if should_swap else agent_1,
        }

        game_env.reset()
        obs, info = game_env.get_initial_observation()
        mask = np.array(info["action_mask"])

        # Needed as you do not need to update in the first step, need both players to actually play something
        prev_rewards = {"player_0": None, "player_1": None}

        while True:
            current_agent = game_env.agent_selection

            if game_env.check_victory(current_agent):
                final_rewards = {}
                for pos in agents.keys():
                    final_rewards[pos] = game_env.rewards[
                        pos
                    ]  # Get the actual final rewards

                # Final update for both agents with terminal state and correct final rewards
                for pos, agent_obj in agents.items():
                    agent_obj.update(final_rewards[pos], None)

                # Track wins for the actual agents
                winning_agent = agents[game_env.agent_selection]

                if winning_agent == agent_0:
                    wins_agent_0 += 1
                else:
                    wins_agent_1 += 1
                break

            agent = agents[current_agent]

            if prev_rewards[current_agent] is not None:
                agent.update(prev_rewards[current_agent], obs)

            action = agent.select_action(obs, mask)
            # print(f"Agent {current_agent} plays {action}")

            game_env.step(action)
            next_obs, reward, termination, truncation, info = game_env.last()
            mask = np.array(info["action_mask"])

            prev_rewards[current_agent] = reward

            obs = next_obs

        if episode % 10 == 0:
            print(f"Episode {episode}")
            print(f"Agent 0 wins: {wins_agent_0}")
            print(f"Agent 1 wins: {wins_agent_1}")
            print(f"Agent 0 played as player_0: {not should_swap}")

    return agent_0, agent_1, wins_agent_0, wins_agent_1


if __name__ == "__main__":
    agent0, agent1, wins_0, wins_1 = play_bluff_game(num_players=2, episodes=200)
    print("\nFinal Results:")
    print(f"Agent 0 wins: {wins_0}")
    print(f"Agent 1 wins: {wins_1}")
    print(f"\nWin Rate Agent 1: {(wins_1 / (wins_0 + wins_1)) * 100:.1f}%")

    # Print Q-table analysis for Agent 0
    print_strategy_analysis(agent0.q_table)

    # Print Q-table analysis for Agent 1
    print_strategy_analysis(agent1.q_table)

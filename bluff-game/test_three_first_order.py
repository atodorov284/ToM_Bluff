import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agents.first_order_dev import FirstOrderAgent
from agents.random_agent import RandomAgent  # noqa: F401
from agents.zero_order import ZeroOrderAgent  # noqa: F401
from envs.bluff_env import env
from utils import print_strategy_analysis


def play_bluff_game(num_players: int = 3, episodes: int = 8) -> None:
    """Play a game of Bluff with the specified number of players."""

    game_env = env(num_players=num_players, render_mode="huma")
    
    agent_1 = FirstOrderAgent(learning_rate=0.1, discount_factor=0.99, epsilon=0.1)
    agent_2 = FirstOrderAgent(learning_rate=0.1, discount_factor=0.99, epsilon=0.1)
    agent_3 = FirstOrderAgent(learning_rate=0.1, discount_factor=0.99, epsilon=0.1)
    
    print(f"Configuration: {agent_1.__class__.__name__}, {agent_2.__class__.__name__}, {agent_3.__class__.__name__}")

    wins_agent_1 = 0
    wins_agent_2 = 0
    wins_agent_3 = 0
    draws = 0

    for episode in range(episodes):
        agents = [agent_1, agent_2, agent_3]

        players = ["player_0", "player_1", "player_2"]

        np.random.shuffle(agents)

        agents = dict(zip(players, agents))

        game_env.reset()
        obs, info = game_env.get_initial_observation()
        mask = np.array(info["action_mask"])

        # Needed as you do not need to update in the first step, need both players to actually play something

        prev_rewards = {"player_0": None, "player_1": None, "player_2": None}
        play = 0
        while True:
            play += 1
            if play >= 1000:
                draws += 1
                break
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

                if winning_agent == agent_1:
                    wins_agent_1 += 1
                elif winning_agent == agent_2:
                    wins_agent_2 += 1
                elif winning_agent == agent_3:
                    wins_agent_3 += 1

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

        # if episode % 10 == 0:
        #     print(f"Episode {episode}")
        #     print(f"Agent 1 wins: {wins_agent_1}")
        #     print(f"Agent 2 wins: {wins_agent_2}")
        #     print(f"Agent 3 wins: {wins_agent_3}")
        #     print(f"Draws: {draws}")

    return agent_1, agent_2, agent_3, wins_agent_1, wins_agent_2, wins_agent_3, draws


def run_experiment(num_runs: int = 50, episodes: int = 10000, seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)

    agent_wins = {
        "wins_agent_1": [],
        "wins_agent_2": [],
        "wins_agent_3": [],
        "draws": [],
    }

    for _ in range(num_runs):
        agent_1, agent_2, agent_3, wins_agent_1, wins_agent_2, wins_agent_3, draws = (
            play_bluff_game(num_players=3, episodes=episodes)
        )

        agent_wins["wins_agent_1"].append(wins_agent_1)
        agent_wins["wins_agent_2"].append(wins_agent_2)
        agent_wins["wins_agent_3"].append(wins_agent_3)
        agent_wins["draws"].append(draws)


    print(
        f"Agent 1 Mean Wins: {np.mean(agent_wins['wins_agent_1'])}, Std: {np.std(agent_wins['wins_agent_1'])}, SEM: {np.std(agent_wins['wins_agent_1']) / np.sqrt(num_runs)}"
    )
    print(
        f"Agent 2 Mean Wins: {np.mean(agent_wins['wins_agent_2'])}, Std: {np.std(agent_wins['wins_agent_2'])}, SEM: {np.std(agent_wins['wins_agent_2']) / np.sqrt(num_runs)}"
    )
    print(
        f"Agent 3 Mean Wins: {np.mean(agent_wins['wins_agent_3'])}, Std: {np.std(agent_wins['wins_agent_3'])}, SEM: {np.std(agent_wins['wins_agent_3']) / np.sqrt(num_runs)}"
    )
    
    print(f"Mean Draws: {np.mean(agent_wins['draws'])}, Std: {np.std(agent_wins['draws'])}, SEM: {np.std(agent_wins['draws']) / np.sqrt(num_runs)}")

    print_strategy_analysis(agent_1.q_table)
    print_strategy_analysis(agent_2.q_table)
    print_strategy_analysis(agent_3.q_table)

if __name__ == "__main__":
    run_experiment(num_runs=100, episodes=10000, seed=1)

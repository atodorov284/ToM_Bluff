import numpy as np
from agents.random_agent import RandomAgent
from agents.zero_order import QLearningAgent
from envs.bluff_env import env


def evaluate_against_random(policy_file, num_episodes=1000):
    # Initialize agents
    qlearn_agent = QLearningAgent(
        learning_rate=0.1, discount_factor=0, epsilon=0
    )  # Set epsilon to 0 for pure exploitation
    policy_file = "agent_0_policy.npy"
    qlearn_agent.load_policy(policy_file)  # Load the trained policy
    random_agent = RandomAgent()

    # Track wins for each agent
    qlearn_wins = 0
    random_wins = 0

    for episode in range(num_episodes):
        # Randomly decide if we should swap agents' positions
        should_swap = np.random.random() < 0.5

        # Assign agents to positions based on swap decision
        agents = {
            "player_0": random_agent if should_swap else qlearn_agent,
            "player_1": qlearn_agent if should_swap else random_agent,
        }

        # Initialize environment
        game_env = env(num_players=2)
        game_env.reset()
        obs, info = game_env.get_initial_observation()

        # Get initial action mask
        mask = (
            np.array(info["action_mask"])
            if "action_mask" in info
            else np.array(obs["action_mask"])
            if isinstance(obs, dict) and "action_mask" in obs
            else np.ones(5, dtype=np.int8)
        )
        mask = np.atleast_1d(mask)

        while True:
            # Get current agent
            current_agent = game_env.agent_selection
            agent = agents[current_agent]

            # Select and take action
            action = agent.select_action(obs, mask)

            # Take step
            game_env.step(action)
            obs, reward, termination, truncation, info = game_env.last()

            # Get mask for next state
            mask = (
                np.array(info["action_mask"])
                if "action_mask" in info
                else np.array(obs["action_mask"])
                if isinstance(obs, dict) and "action_mask" in obs
                else np.ones(5, dtype=np.int8)
            )
            mask = np.atleast_1d(mask)

            # Check if game is over
            if termination or truncation:
                # Track wins based on final rewards
                winning_agent = agents[game_env.agent_selection]
                if winning_agent == qlearn_agent:
                    qlearn_wins += 1
                else:
                    random_wins += 1
                break

        # Print progress
        if (episode + 1) % 1000 == 0:
            print(f"\nEpisode {episode + 1}")
            print(f"Q-Learning Agent wins: {qlearn_wins}")
            print(f"Random Agent wins: {random_wins}")
            print(f"Q-Learning Agent win rate: {qlearn_wins/(episode + 1):.2%}")

    return qlearn_wins, random_wins


if __name__ == "__main__":
    # You can replace this with the path to your trained policy file
    policy_file = "agent_0_policy.npy"

    print("Evaluating Q-Learning agent against Random agent...")
    q_wins, rand_wins = evaluate_against_random(policy_file, num_episodes=10000)

    print("\nFinal Results:")
    print(f"Q-Learning Agent wins: {q_wins}")
    print(f"Random Agent wins: {rand_wins}")
    print(f"Q-Learning Agent win rate: {q_wins/(q_wins + rand_wins):.2%}")

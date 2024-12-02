import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.bluff_env import env
from agents.zero_order import QLearningAgent

import numpy as np

def play_bluff_game(num_players=2, episodes=100):
    # Initialize agents
    agent_0 = QLearningAgent(learning_rate=0.1, discount_factor=1, epsilon=0.1)  # removed discount
    agent_1 = QLearningAgent(learning_rate=0.1, discount_factor=1, epsilon=0.1)  # removed discount
    
    # Track wins for each agent (not position)
    wins_agent_0 = 0
    wins_agent_1 = 0
    
    for episode in range(episodes):
        # Randomly decide if we should swap agents
        should_swap = np.random.random() < 0.5
        
        # Assign agents to positions based on swap decision
        agents = {
            'player_0': agent_1 if should_swap else agent_0,
            'player_1': agent_0 if should_swap else agent_1
        }
        
        # Initialize environment
        game_env = env(num_players=num_players)
        prev_rewards = {'player_0': None, 'player_1': None}
        
        game_env.reset()
        obs, info = game_env.get_initial_observation()

        # Get initial action mask
        mask = (np.array(info["action_mask"]) if "action_mask" in info
                else np.array(obs["action_mask"]) if isinstance(obs, dict) and "action_mask" in obs
                else np.ones(5, dtype=np.int8))
        mask = np.atleast_1d(mask)

        while True:
            # Get current agent
            current_agent = game_env.agent_selection
            agent = agents[current_agent]

            #print current turn
            #print(f"Current turn: {current_agent}")
            
            # Update Q-values at the start of agent's turn if we have a previous reward
            if prev_rewards[current_agent] is not None:
                agent.update(prev_rewards[current_agent], obs)
            
            # Select and take action
            action = agent.select_action(obs, mask)
            
            # Take action
            game_env.step(action)
            next_obs, reward, termination, truncation, info = game_env.last()
            
            # Store reward for next turn
            prev_rewards[current_agent] = reward
            
            # Get mask for next state
            mask = (np.array(info["action_mask"]) if "action_mask" in info
                   else np.array(next_obs["action_mask"]) if isinstance(next_obs, dict) and "action_mask" in next_obs
                   else np.ones(5, dtype=np.int8))
            mask = np.atleast_1d(mask)

            # Check if game is over
            if termination or truncation:
                # Get final rewards for both agents
                final_rewards = {}
                for pos in agents.keys():
                    final_rewards[pos] = game_env.rewards[pos]  # Get the actual final rewards

                # Final update for both agents with terminal state and correct final rewards
                for pos, agent_obj in agents.items():
                    agent_obj.update(final_rewards[pos], None)
                
                # Track wins for the actual agents
                winning_agent = agents[game_env.agent_selection]
                
                # print hands of both agents
                #print(f"Agent 0: {game_env.player_hands['player_0']}")
                #print(f"Agent 1: {game_env.player_hands['player_1']}")

                if winning_agent == agent_0:
                    wins_agent_0 += 1
                else:
                    wins_agent_1 += 1
                break
            
            obs = next_obs
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}")
            print(f"Agent 0 wins: {wins_agent_0}")
            print(f"Agent 1 wins: {wins_agent_1}")
            print(f"Agent 0 played as player_0: {not should_swap}")

        # Decay epsilon
        if episode % 1000 == 0 and episode > 0:
            agent_0.epsilon *= 0.95
            agent_1.epsilon *= 0.95
    
    # Save the trained policies
    agent_0.save_policy("agent_0_policy.npy")
    agent_1.save_policy("agent_1_policy.npy")
    
    return agent_0, agent_1, wins_agent_0, wins_agent_1

if __name__ == "__main__":
    agent1, agent2, wins_0, wins_1 = play_bluff_game(num_players=2, episodes=1000)
    print("\nFinal Results:")
    print(f"Agent 0 wins: {wins_0}")
    print(f"Agent 1 wins: {wins_1}")
    print("\nFinal Q-values:")
    #print("Agent 1:", agent1.q_table, "states")
    #print("Agent 2:", agent2.q_table, "states")
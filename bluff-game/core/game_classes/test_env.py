from environment import env

import random


def play_bluff_game(num_players=2):
    # Initialize the Bluff environment
    game_env = env(num_players=num_players, render_mode="human")
    
    # Reset the environment to start the game
    game_env.reset()

    print("\n--- Starting the Bluff Game ---\n")
    
    # Run the game loop
    while not all(game_env.terminations.values()):
        # Get the current agent
        agent = game_env.agent_selection

        # Observe the environment state for the current agent
        
        # Decide on an action (play or challenge)
        # Random decision for demonstration
        action = game_env.action_space(agent).sample() # ACTION_PLAY = 0, ACTION_CHALLENGE = 1. First action is always 0.
        # Take the step
        game_env.step(action)



    # Print the final rewards
    print("\n--- Final Results ---")
    print(f"Final Rewards: {game_env.rewards}")
    game_env.close()



play_bluff_game(num_players=3)

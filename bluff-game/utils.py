def print_strategy_analysis(q_table: dict) -> None:
    """Print strategy distribution analysis of the Q-table."""
    # Analyze preference for different types of actions
    action_type_counts = {
        "Challenge": 0,
        "Truth": 0,
        "Bluff": 0
    }
    
    # Analyze preferences based on pile size
    pile_meanings = {
        0: "Small (0-2)",
        1: "Medium (3-6)",
        2: "Large (7+)"
    }
    
    pile_size_preferences = {
        0: {"Challenge": 0, "Truth": 0, "Bluff": 0},
        1: {"Challenge": 0, "Truth": 0, "Bluff": 0},
        2: {"Challenge": 0, "Truth": 0, "Bluff": 0}
    }
    
    for state in q_table:
        best_action = q_table[state].argmax()
        pile_size = state[4]  # Get pile size from state tuple
        
        # Categorize the action
        if best_action == 0:
            action_type = "Challenge"
        elif 1 <= best_action <= 4:
            action_type = "Truth"
        else:
            action_type = "Bluff"
            
        action_type_counts[action_type] += 1
        pile_size_preferences[pile_size][action_type] += 1
    
    total_states = len(q_table)
    if total_states > 0:
        print("\nOverall Strategy Distribution:")
        print("-" * 40)
        for action_type, count in action_type_counts.items():
            percentage = (count / total_states) * 100
            print(f"{action_type:10}: {percentage:5.1f}%")
            
        print("\nStrategy Distribution by Pile Size:")
        print("-" * 40)
        for pile_size, counts in pile_size_preferences.items():
            print(f"\n{pile_meanings[pile_size]}:")
            pile_total = sum(counts.values())
            if pile_total > 0:
                for action_type, count in counts.items():
                    percentage = (count / pile_total) * 100
                    print(f"{action_type:10}: {percentage:5.1f}%")

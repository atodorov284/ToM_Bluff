import numpy as np

"""
This file contains the code to count the total number of valid combinations of actions as well as tests for the mask applied to the actions
"""


def is_valid_sum(nums, total=4):
    return sum(nums) <= total


def count_valid_combinations():
    cards_played_from_rank = 0
    cards_left_to_play = 4 - cards_played_from_rank
    mask = []
    for a in range(cards_left_to_play + 1):
        for b in range(cards_left_to_play + 1):
            for c in range(cards_left_to_play + 1):
                for d in range(cards_left_to_play + 1):
                    if is_valid_sum([a, b, c, d], cards_left_to_play):
                        mask.append([a, b, c, d])
    return mask


def numpy_test():
    player_hands = [0, 1, 1, 1]
    cards_played_from_rank = 3
    cards_left_to_play = 4 - cards_played_from_rank
    cards_left_to_play = min(cards_left_to_play, sum(player_hands))
    mask = []

    num_of_aces_in_hand = player_hands[0]
    num_of_jacks_in_hand = player_hands[1]
    num_of_queens_in_hand = player_hands[2]
    num_of_kings_in_hand = player_hands[3]
    a, b, c, d = np.indices(
        (
            min(cards_left_to_play + 1, num_of_aces_in_hand + 1),
            min(cards_left_to_play + 1, num_of_jacks_in_hand + 1),
            min(cards_left_to_play + 1, num_of_queens_in_hand + 1),
            min(cards_left_to_play + 1, num_of_kings_in_hand + 1),
        )
    )

    # Flatten the arrays to create all combinations
    combinations = np.stack([a.ravel(), b.ravel(), c.ravel(), d.ravel()], axis=-1)

    valid_combinations = combinations[
        np.sum(combinations, axis=1) <= cards_left_to_play
    ]

    # Convert to list if needed
    mask = valid_combinations.tolist()
    return mask


# Count total valid combinations
#result = numpy_test()
#print(f"Total number of valid combinations: {result}")
#print(f"Total number of valid combinations: {len(result)}")

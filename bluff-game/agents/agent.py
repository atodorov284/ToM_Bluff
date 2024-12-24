from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, action_space: np.ndarray, mask: np.ndarray) -> int:
        """
        Abstract method to select an action based on the action space and the mask.
        action_space: A probability distribution or set of probabilities for each action.
        mask: A binary mask of valid actions.
        """
        pass

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        """
        Update the agent's knowledge based on the action taken and the reward received.
        Args:
            action: The arm that was pulled.
            reward: The reward received after pulling the arm.
        """
        pass

    def _discretize_state(self, observation: dict) -> tuple:
        """Convert the observation dictionary into a hashable state tuple."""
        # Extract the hand frequency vector
        hand_freq = tuple(observation["hand"])  # Already in the format we want

        # Discretize pile size into 3 categories
        pile_size = observation["central_pile_size"]
        if pile_size <= 4:
            discrete_pile = 0  # small
        elif pile_size <= 8:
            discrete_pile = 1  # medium
        else:
            discrete_pile = 2  # large

        # Get the current rank (already discrete 0-3)
        current_rank = observation["current_rank"]

        cards_last_played = observation["cards_other_agent_played"]

        # Return state tuple
        return (*hand_freq, discrete_pile, current_rank, cards_last_played)

    def _action_to_index(self, action: list) -> int:
        """Convert action to index in Q-table."""
        return action[0] * 125 + action[1] * 25 + action[2] * 5 + action[3]

    def _index_to_action(self, index: int) -> list:
        """Convert index in Q-table to action."""
        return [index // 125, (index % 125) // 25, (index % 25) // 5, index % 5]

    def _get_state(self, observation: dict) -> tuple:
        """Convert observation into state tuple (cards_of_curr_rank, other_cards, current_rank)."""
        hand_freq = observation["hand"]
        current_rank = observation["current_rank"]

        cards_of_curr_rank = hand_freq[current_rank]
        other_cards = sum(hand_freq) - cards_of_curr_rank

        num_cards_other_agent_played = observation["cards_other_agent_played"]

        pile_size = observation["central_pile_size"]
        
        cards_in_other_agent_hand = observation["cards_in_other_agent_hand"]
        
        if cards_in_other_agent_hand < 3:
            discrete_num_cards_other = 0
        elif cards_in_other_agent_hand > 3 and cards_in_other_agent_hand < 7:
            discrete_num_cards_other = 1
        else:
            discrete_num_cards_other = 2

        # discretize pile size
        if pile_size < 3:
            discrete_pile = 0
        elif pile_size > 3 and pile_size < 7:
            discrete_pile = 1
        else:
            discrete_pile = 2

        return (
            cards_of_curr_rank,
            other_cards,
            current_rank,
            num_cards_other_agent_played,
            discrete_pile,
            discrete_num_cards_other
        )

    def _convert_action_to_full(self, action: int, hand_freq: list) -> list:
        """Convert action number to full action format.
        0: Challenge
        1-4: Play 1-4 truthful cards
        5-8: Play 1-4 bluff cards
        """
        if action == 0:  # Challenge
            return self.ACTION_CHALLENGE

        full_action = [0, 0, 0, 0]
        if action <= 4:  # Truthful play
            num_cards = action
            full_action[self.current_rank] = num_cards
        else:  # Bluff
            num_cards = action - 4

            # Get cards from other ranks
            other_ranks = [i for i in range(4) if i != self.current_rank]
            available_cards = [
                (rank, hand_freq[rank]) for rank in other_ranks if hand_freq[rank] > 0
            ]
            if not available_cards:
                return self.ACTION_CHALLENGE  # Fallback if we can't bluff

            # Sort ranks by priority:
            # 1. Ranks that have already passed (lower than current)
            # 2. Remaining ranks
            available_cards.sort(
                key=lambda x: (
                    0 if x[0] < self.current_rank else 1,  # Prioritize past ranks
                    -x[1],  # Then by number of cards (most cards first)
                )
            )

            # Use cards from multiple ranks if needed
            cards_needed = num_cards
            for rank, count in available_cards:
                if cards_needed <= 0:
                    break
                cards_to_use = min(count, cards_needed)
                full_action[rank] = cards_to_use
                cards_needed -= cards_to_use

        return full_action

    def _get_valid_actions(self, mask: list, hand_freq: list) -> list:
        """Get list of valid actions in our simplified action space."""
        valid_actions = []

        # Check if challenge is valid
        if any(np.array_equal(action, self.ACTION_CHALLENGE) for action in mask):
            valid_actions.append(0)

        # For each action in mask, determine if it corresponds to a valid play in our action space
        for mask_action in mask:
            if np.array_equal(mask_action, self.ACTION_CHALLENGE):
                continue

            max_cards = sum(mask_action)  # Maximum cards allowed by this mask action

            # Check if this is a truth play (only uses current rank)
            if (
                mask_action[self.current_rank] > 0
            ):  # If we can play any cards of current rank
                max_truth_cards = mask_action[
                    self.current_rank
                ]  # How many we can play truthfully
                for num_cards in range(1, max_truth_cards + 1):
                    if hand_freq[self.current_rank] >= num_cards:
                        valid_actions.append(num_cards)  # Actions 1-4

            # Check if this could be a bluff play (uses other ranks)
            if mask_action[self.current_rank] == 0:
                # Calculate available cards in the required ranks
                available_cards = sum(
                    min(mask_action[i], hand_freq[i])
                    for i in range(4)
                    if i != self.current_rank
                )
                # Can play any number of cards up to min(max_cards, available_cards)
                for num_cards in range(1, min(max_cards, available_cards) + 1):
                    valid_actions.append(num_cards + 4)  # Actions 5-8

        return sorted(list(set(valid_actions)))

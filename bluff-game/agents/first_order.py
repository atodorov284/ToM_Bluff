from agents.agent import BaseAgent
from agents.zero_order import QLearningAgent


class FirstOrderAgent(BaseAgent):
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
    ) -> None:
        """Initialize the first-order ToM agent."""
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        self.last_state = None
        self.last_action = None
        self.ACTION_CHALLENGE = [0, 0, 0, 0]
        self.NUM_ACTIONS = 9  # challenge + 4 truth + 4 bluff
        opponent_model = QLearningAgent(learning_rate, discount_factor, epsilon)

    def select_action(self, observation: dict, mask: list):
        """
                If else statemment.
                If you believe opponent is bluffing, then challenge,

                Else figure out whether opponent will challenge you and play bluff ofr truthfully accordingly.

        """

        pass

    def update(self, reward: float, next_observation: dict = None) -> None:
        pass

    def _update_opponent_model(self, reward: float, next_observation: dict) -> list:
        pass

    def _select_opponent_action(self, observation: dict, mask: list) -> list:
        pass

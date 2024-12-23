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

    def select_action(self, observation: dict, mask: list):
        """
        FOR DECIDING WHETHER TO CHALLENGE OR NOT
                If you believe opponent is bluffing, then increase the q value for challenging them.
                if they dont play the number of cards you believe, then fall back to using the q table to figure out if you should challenge them.
                Else figure out whether opponent will challenge you and play bluff or truthfully accordingly.
                
        FOR DECIDING WHETHER TO BLUFF OR PLAY TRUTHFULLY
                Do it iteratively. Simulate possible actions you can make and using the model estimate what the opponent would do. An action is selected if it is valid.
                Valid actions in this case are for example if you simulate playing bluff, and you think your opponent would not challenge you, or if you played truthfully 
                and you think your opponent would challenge you. Essentially whenever you would trick your opponent you have a valid action and the action you actually play is 
                the valid action that you estimate to bring you the most reward.        
                
        """

        pass

    def update(self, reward: float, next_observation: dict = None) -> None:
        pass

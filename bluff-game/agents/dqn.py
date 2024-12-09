import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from .agent import BaseAgent

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in indices))
        
        # Replace None with zero tensors
        next_states = torch.stack([
            torch.zeros_like(torch.tensor(states[0], dtype=torch.float32))
            if ns is None else torch.tensor(ns, dtype=torch.float32)
            for ns in next_states
        ])
        
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            next_states,  # Already stacked into a tensor
            torch.tensor(dones, dtype=torch.float32),
        )

        

    def __len__(self):
        return len(self.buffer)



class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ApproxQLearningAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, learning_rate=0.001, discount_factor=0.95, epsilon=0.1, 
                 buffer_capacity=10000, batch_size=64, target_update_freq=100):
        """Initialize the Approximation-Based Q-Learning Agent."""
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Policy and target networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.last_state = None
        self.last_action = None
        self.step_counter = 0

    def select_action(self, observation: dict, mask: list) -> int:
        """Select action using epsilon-greedy policy."""
        state = self._discretize_state(observation)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        if np.random.random() < self.epsilon:
            # Random valid action
            mask_indices = [self._action_to_index(valid_action) for valid_action in mask]
            random_action_index = np.random.choice(mask_indices)
            action = self._index_to_action(random_action_index)
        else:
            # Greedy action selection
            q_values = deepcopy(self.q_network(state_tensor).detach().numpy())
            mask_indices = [self._action_to_index(valid_action) for valid_action in mask]
            invalid_actions = [i for i in range(self.action_dim) if i not in mask_indices]
            q_values[0][invalid_actions] = -np.inf
            action = self._index_to_action(np.argmax(q_values[0]))

        self.last_state = state
        self.last_action = self._action_to_index(action)
        return action

    def update(self, reward: float, next_observation: dict = None) -> None:
        """Update Q-values using function approximation."""
        if self.last_state is None or self.last_action is None:
            return

        # Add experience to replay buffer
        next_state = self._discretize_state(next_observation) if next_observation else None
        done = next_observation is None
        self.replay_buffer.push(self.last_state, self.last_action, reward, next_state, done)

        # Skip training if replay buffer is too small
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Compute targets
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute loss and update network
        loss = self.loss_fn(current_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network weights periodically
        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

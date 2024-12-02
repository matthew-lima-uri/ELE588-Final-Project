import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import random
import numpy as np

from Agent import Agent
from DQN_Utils import NStepReplayBuffer


# Implements a DQN network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        # Define network architecture
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, action_size)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)

# Implements a dueling DQN network
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()

        # Shared layers
        self.fc1 = nn.Linear(state_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)

        # Value stream
        self.value_fc = nn.Linear(512, 256)
        self.value_out = nn.Linear(256, 1)

        # Advantage stream
        self.advantage_fc = nn.Linear(512, 256)
        self.advantage_out = nn.Linear(256, action_size)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))

        # Value stream
        value = F.leaky_relu(self.value_fc(x))
        value = self.value_out(value)

        # Advantage stream
        advantage = F.leaky_relu(self.advantage_fc(x))
        advantage = self.advantage_out(advantage)

        # Combine streams
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

# Define the DQN Agent
class DoubleDQNAgent(Agent):
    def __init__(self, state_size_in, action_size_out, n_step=1, gamma=0.99, memory=None, max_pot=10000,
                 target_update_freq=10):
        self.state_size = state_size_in
        self.action_size = action_size_out
        self.__gamma = gamma  # Discount factor
        if memory is None:
            self.__memory = NStepReplayBuffer(100000, n_step, gamma)
        else:
            self.__memory = memory
        self.__n_step = n_step
        self.__epsilon = 1.0  # Exploration rate
        self.__epsilon_min = 0.1
        self.__epsilon_decay = 0.99995
        self.__learning_rate = 0.0005
        self.__batch_size = 128
        self.__epoch_reward = 0
        self.__update_steps = 0
        self.__target_update_frequency = target_update_freq
        self.__loss_fn = nn.SmoothL1Loss()
        self.__max_pot = max_pot

        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.__device = torch.device("cpu")

        self.__policy_net = DQN(state_size_in, action_size_out).to(self.__device)
        self.__target_net = DQN(state_size_in, action_size_out).to(self.__device)
        self.update_target_network()
        self.__optimizer = optim.AdamW(self.__policy_net.parameters(), lr=self.__learning_rate)
        self.__scheduler = torch.optim.lr_scheduler.StepLR(self.__optimizer, gamma=0.01, step_size=self.__target_update_frequency)

    def update_target_network(self):
        self.__target_net.load_state_dict(self.__policy_net.state_dict())

    def act(self, state, greedy=False):
        # Epsilon-greedy action selection
        if not greedy:
            if np.random.rand() <= self.__epsilon:
                return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.__device)

        # Set the model to evaluation mode
        self.__policy_net.eval()
        with torch.no_grad():
            q_values = self.__policy_net(state)
        # Set the model back to training mode
        self.__policy_net.train()

        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.__memory.push((state, action, reward, next_state, done))

    def replay(self):
        if len(self.__memory) < self.__batch_size:
            return

        batch = self.__memory.sample(self.__batch_size)

        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).to(self.__device, non_blocking=True)
        actions = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.int64).unsqueeze(1).to(self.__device, non_blocking=True)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1).to(self.__device, non_blocking=True)
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32).to(self.__device, non_blocking=True)
        dones = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float32).unsqueeze(1).to(self.__device, non_blocking=True)

        # Compute current Q-values
        current_q_values = self.__policy_net(states).gather(1, actions)

        with torch.no_grad():
            # Double DQN target calculation
            # Action selection with policy network
            next_actions = self.__policy_net(next_states).argmax(dim=1, keepdim=True)
            # Action evaluation with target network
            next_q_values = self.__target_net(next_states).gather(1, next_actions)
            # Compute target Q-values
            target_q_values = rewards + (self.__gamma ** self.__n_step) * next_q_values * (1 - dones)

        # Compute loss
        loss = self.__loss_fn(current_q_values, target_q_values)

        # Optimize the model
        self.__optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.__policy_net.parameters(), max_norm=10.0)
        self.__optimizer.step()

        self.__update_steps += 1
        if self.__update_steps % self.__target_update_frequency == 0:
            self.update_target_network()

        # Decay epsilon
        if self.__epsilon > self.__epsilon_min:
            self.__epsilon *= self.__epsilon_decay

        # Update the learning rate scheduler
        self.__scheduler.step()

        return loss.item()

    def update_epoch_reward(self, new_reward):
        self.__epoch_reward = self.__epoch_reward + new_reward

    def get_epoch_reward(self):
        return self.__epoch_reward

    def reset_epoch_reward(self):
        self.__epoch_reward = 0

    def get_policy_network(self):
        return self.__policy_net

    def load_policy_network(self, filepath):
        self.__policy_net.load_state_dict(
            torch.load(filepath, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        self.update_target_network()

    def get_memory(self):
        return self.__memory

    def get_epsilon(self):
        return self.__epsilon

# Define the Dueling DQN Agent
class DuelingDQNAgent(Agent):
    def __init__(self, state_size_in, action_size_out, n_step=1, gamma=0.99, memory=None, max_pot=10000,
                 target_update_freq=10, learning_rate_step=10):
        self.state_size = state_size_in
        self.action_size = action_size_out
        self.__gamma = gamma  # Discount factor
        if memory is None:
            self.__memory = NStepReplayBuffer(50000, n_step, gamma)
        else:
            self.__memory = memory
        self.__n_step = n_step
        self.__epsilon = 1.0  # Exploration rate
        self.__epsilon_min = 0.1
        self.__epsilon_decay = 0.99995
        self.__learning_rate = 0.0005
        self.__batch_size = 128
        self.__epoch_reward = 0
        self.__update_steps = 0
        self.__target_update_frequency = target_update_freq
        self.__loss_fn = nn.SmoothL1Loss()
        self.__max_pot = max_pot

        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.__device = torch.device("cpu")

        self.__policy_net = DuelingDQN(state_size_in, action_size_out).to(self.__device)
        self.__target_net = DuelingDQN(state_size_in, action_size_out).to(self.__device)
        self.update_target_network()
        self.__optimizer = optim.AdamW(self.__policy_net.parameters(), lr=self.__learning_rate)
        self.__scheduler = torch.optim.lr_scheduler.StepLR(self.__optimizer, learning_rate_step)

    def update_target_network(self):
        tau = 0.01  # Soft update parameter
        for target_param, policy_param in zip(self.__target_net.parameters(), self.__policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def act(self, state, greedy=False):
        # Epsilon-greedy action selection
        if not greedy:
            if np.random.rand() <= self.__epsilon:
                return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.__device)

        # Set the model to evaluation mode
        self.__policy_net.eval()
        with torch.no_grad():
            q_values = self.__policy_net(state)
        # Set the model back to training mode
        self.__policy_net.train()

        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.__memory.push((state, action, reward, next_state, done))

    def replay(self):
        if len(self.__memory) < self.__batch_size:
            return

        batch = self.__memory.sample(self.__batch_size)

        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).to(self.__device, non_blocking=True)
        actions = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.int64).unsqueeze(1).to(self.__device, non_blocking=True)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1).to(self.__device, non_blocking=True)
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32).to(self.__device, non_blocking=True)
        dones = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float32).unsqueeze(1).to(self.__device, non_blocking=True)

        # Compute current Q-values using policy network
        current_q_values = self.__policy_net(states).gather(1, actions)

        with torch.no_grad():
            # Double DQN target calculation
            # Action selection with policy network
            next_actions = self.__policy_net(next_states).argmax(dim=1, keepdim=True)
            # Action evaluation with target network
            next_q_values = self.__target_net(next_states).gather(1, next_actions)
            # Compute target Q-values
            target_q_values = rewards + (self.__gamma ** torch.linspace(1, self.__batch_size, self.__batch_size, device=self.__device).reshape(self.__batch_size, 1)) * next_q_values

        # Compute loss
        loss = self.__loss_fn(current_q_values, target_q_values)

        # Optimize the model
        self.__optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.__policy_net.parameters(), max_norm=5.0)
        self.__optimizer.step()

        # Update target network and learning rate periodically
        self.__update_steps += 1
        if self.__update_steps % self.__target_update_frequency == 0:
            self.update_target_network()
            self.__scheduler.step()

        # Decay epsilon
        if self.__epsilon > self.__epsilon_min:
            self.__epsilon *= self.__epsilon_decay

        return loss.item()

    def update_epoch_reward(self, new_reward):
        self.__epoch_reward = self.__epoch_reward + new_reward

    def get_epoch_reward(self):
        return self.__epoch_reward

    def reset_epoch_reward(self):
        self.__epoch_reward = 0

    def get_policy_network(self):
        return self.__policy_net

    def load_policy_network(self, filepath):
        self.__policy_net.load_state_dict(
            torch.load(filepath, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        self.update_target_network()

    def get_memory(self):
        return self.__memory

    def get_epsilon(self):
        return self.__epsilon
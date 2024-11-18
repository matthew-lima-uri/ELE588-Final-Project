import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from gym import Environment

"""
System constants

State Size: Defines the input space for the model. Derived from own cards, own bank, community cards, current pool, 
betting round, other player states, other player actions, other player bets, other player banks.

Action Size: Defines the output space for the model. Derived from the possible actions and raise amount.
"""
state_size = (2*2) + 1 + (5*2) + 1 + 1 + (5*1) + (5*1) + (5*1) + (5*1)
action_size = 2

# Define the Deep Q-Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Define the Agent
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(10000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.batch_size = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.update_target_network()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(np.array([m[0] for m in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([m[1] for m in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([m[2] for m in minibatch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([m[3] for m in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([m[4] for m in minibatch])).unsqueeze(1).to(self.device)

        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        # Compute next Q values from target network
        max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        # Compute expected Q values
        expected_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Compute loss
        loss = nn.MSELoss()(current_q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class Card:

    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def map_to_neurons(self):
        return [self.rank, self.suit]

# Function to map PokerKit card definitions to Agent definitions
def get_card(card):
    rank_dict = {'?':0, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'T':10,
                 'J':11, 'Q':12, 'K':13, 'A':14}
    suit_dict = {'h':1, 'd':2, 'c':3, 's':4}

    rank = rank_dict.get(card[0])
    suit = suit_dict.get(card[1])

    return Card(rank, suit)

# Function to convert the state from the environment into a neural network input
def get_state_vector(state):

    # Initialize the state vector
    state_vector = np.zeros(state_size)

    # Encode private cards


    # Encode own bank

    # Encode community cards

    # Encode the pool

    # Encode the betting round

    # Encode the other player states

    # Encode the other player actions

    # Encode the other player banks

    # Combine card vector and scalar features

    return state_vector


# Main training loop
def main():

    # Define training parameters
    training_iterations = 1000

    # Initialize game environment
    agent = Agent(state_size, action_size)
    env = Environment(number_of_players=6) # Agent + 5 other players

    update_target_every = 100  # Update target network every 100 episodes

    for episode in range(training_iterations):
        # Reset the environment
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Get the game state
            state = env.get_game_state()
            # Get the state vector from the game state
            state_vector = get_state_vector(state)
            # Agent takes action
            action = agent.act(state_vector)
            # Map action index to actual action in the environment
            next_state = env.execute_agent_action(action)
            # Update state vector based on the new state from the agent's action
            next_state_vector = get_state_vector(next_state)
            # Get the reward from the environment based on the previous action
            reward = env.get_reward()
            # Remember the experience
            agent.remember(state_vector, action, reward, next_state_vector, done)
            # Learn from experience
            agent.replay()
            # Move to the next state
            state = next_state
            total_reward += reward

        # Update the target network periodically
        if episode % update_target_every == 0:
            agent.update_target_network()
            print(f"Episode {episode}, Total Reward: {total_reward}, "
                  f"Epsilon: {agent.epsilon:.2f}")

    # Save the trained model
    torch.save(agent.policy_net.state_dict(), 'dqn_poker_model.pth')

if __name__ == "__main__":
    main()
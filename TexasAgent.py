from copy import deepcopy

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from gym import PokerGame
import time
import sys
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

"""
System constants

State Size: Defines the input space for the model. Derived from own cards, own bank, community cards, current pot, 
betting round, other player statuses, other player bets, other player banks.

Action Size: Defines the output space for the model. Derived from the possible actions (Fold, Call, Raise, All-in)
"""
number_of_agents = 6
state_size = (2 * 2) + 1 + (5 * 2) + 1 + 1 + ((number_of_agents - 1) * 1) + ((number_of_agents - 1) * 1) + (
            (number_of_agents - 1) * 1)
action_size = 4


# Define the Deep Q-Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        # Define network architecture
        self.fc1 = nn.Linear(state_size, 512)
        self.rl1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(512, 256)
        self.rl1 = nn.LeakyReLU()
        self.fc3 = nn.Linear(256, 256)
        self.rl1 = nn.LeakyReLU()
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


# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_buffer(self):
        return self.memory

    def copy(self, memory_buffer):
        self.memory = deepcopy(memory_buffer)

    def __len__(self):
        return len(self.memory)


class NStepReplayBuffer:
    def __init__(self, capacity, n_step, gamma):
        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def push(self, experience):
        self.n_step_buffer.append(experience)
        if len(self.n_step_buffer) == self.n_step:
            state, action, _, _, _ = self.n_step_buffer[0]
            reward, next_state, done = self._get_n_step_info()
            self.buffer.append((state, action, reward, next_state, done))

    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][2:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            _, _, r, ns, d = transition  # Correct unpacking
            reward = r + self.gamma * reward * (1 - d)
            if d:
                next_state = ns
                done = d
        return reward, next_state, done

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def get_buffers(self):
        return self.buffer, self.n_step_buffer

    def copy(self, memory_buffer, n_step_buffer):
        self.buffer = deepcopy(memory_buffer)
        self.n_step_buffer = deepcopy(n_step_buffer)

    def __len__(self):
        return len(self.buffer)


# Define the Agent
class Agent:
    def __init__(self, state_size, action_size, n_step=1, gamma=0.99, memory=None, max_pot=10000,
                 target_update_freq=10):
        self.state_size = state_size
        self.action_size = action_size
        self.__gamma = gamma  # Discount factor
        if memory is None:
            self.__memory = NStepReplayBuffer(1000000, n_step, gamma)
        else:
            self.__memory = memory
        self.__n_step = n_step
        self.__epsilon = 1.0  # Exploration rate
        self.__epsilon_min = 0.01
        self.__epsilon_decay = 0.9999
        self.__learning_rate = 0.01
        self.__batch_size = 512
        self.__epoch_reward = 0
        self.__update_steps = 0
        self.__target_update_frequency = target_update_freq
        self.__loss_fn = nn.SmoothL1Loss()
        self.__max_pot = max_pot

        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__policy_net = DQN(state_size, action_size).to(self.__device)
        self.__target_net = DQN(state_size, action_size).to(self.__device)
        self.update_target_network()
        self.__optimizer = optim.AdamW(self.__policy_net.parameters(), lr=self.__learning_rate)
        self.__scheduler = torch.optim.lr_scheduler.StepLR(self.__optimizer, step_size=self.__target_update_frequency)

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

    @staticmethod
    def decode_action(action):

        # All-in
        if action == 0:
            return "A"
        # Fold
        elif action == 1:
            return "F"
        # Raise
        elif action == 2:
            return "R"
        # Call/Check
        elif action == 3:
            return "C"
        # Default to calling
        else:
            return "C"

    @staticmethod
    def encode_action(decoded_action):
        # All-in
        if decoded_action == "A":
            return 0
        # Fold
        elif decoded_action == "F":
            return 1
        # Raise
        elif decoded_action == "R":
            return 2
        # Call/Check
        elif decoded_action == "C":
            return 3
        # Default to calling
        else:
            return 3

    def remember(self, state, action, reward, next_state, done):
        self.__memory.push((state, action, reward, next_state, done))

    def replay(self):
        if len(self.__memory) < self.__batch_size:
            return None

        batch = self.__memory.sample(self.__batch_size)

        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).to(self.__device)
        actions = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.int64).unsqueeze(1).to(self.__device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1).to(self.__device)
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32).to(self.__device)
        dones = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float32).unsqueeze(1).to(self.__device)

        # Compute current Q-values
        current_q_values = self.__policy_net(states).gather(1, actions)

        # Compute target Q-values using n-step returns
        with torch.no_grad():
            next_q_values = self.__target_net(next_states).max(1)[0].unsqueeze(1)
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

        return loss

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

    @staticmethod
    # Function to map PokerKit card definitions to Agent definitions
    def get_card(card):
        rank_dict = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10,
                     'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        suit_dict = {'h': 1, 'd': 2, 'c': 3, 's': 4}

        return rank_dict[card[0]], suit_dict[card[1]]

    # Function to convert the state from the environment into a neural network input
    def get_state_vector(self, state, player_index):

        # Initialize the state vector
        state_vector = np.zeros(state_size)

        # Encode private cards
        if not (state.player_cards[player_index] is None):
            state_vector[0: 2] = self.get_card(state.player_cards[player_index][0])
            state_vector[2: 4] = self.get_card(state.player_cards[player_index][1])
            state_vector[0] = state_vector[0] / 14  # Max value for a rank
            state_vector[1] = state_vector[1] / 4  # Max value for a suit
            state_vector[2] = state_vector[0] / 14  # Max value for a rank
            state_vector[3] = state_vector[1] / 4  # Max value for a suit

        # Encode own bank
        state_vector[4] = state.player_banks[player_index] / self.__max_pot  # Max value for a bank

        # Encode community cards
        for i in range(len(state.board_cards)):
            if not (state.board_cards[i] is None):
                state_vector[5 + (i * 2): 5 + (i * 2) + 2] = self.get_card(state.board_cards[i])
                state_vector[5 + (i * 2)] = state_vector[5 + (i * 2)] / 14  # Max value for a rank
                state_vector[5 + (i * 2) + 1] = state_vector[5 + (i * 2) + 1] / 4  # Max value for a suit
                # Community cards that have an unknown value will remain 0

        # Encode the pool
        state_vector[15] = state.pot / (self.__max_pot * number_of_agents)  # Maximum theoretical pot

        # Encode the betting round
        state_vector[16] = state.betting_round / 4  # Max value for betting rounds

        # Encode the other player statuses
        agent_index = 0
        for i in range(number_of_agents):
            # Skip the own players status
            if i == player_index:
                continue
            state_vector[17 + agent_index] = state.player_statuses[agent_index]  # Already capped between 0 and 1
            agent_index = agent_index + 1

        # Encode the other player bets
        agent_index = 0
        for i in range(number_of_agents):
            # Skip the own players status
            if i == player_index:
                continue
            state_vector[22 + agent_index] = state.player_bets[
                                                 agent_index] / self.__max_pot  # Theoretical max bet per player
            agent_index = agent_index + 1

        # Encode the other player banks
        agent_index = 0
        for i in range(number_of_agents):
            # Skip the own players bank
            if i == player_index:
                continue
            state_vector[27 + agent_index] = state.player_banks[agent_index] / self.__max_pot  # Max bank per player
            agent_index = agent_index + 1

        return state_vector

    def get_epsilon(self):
        return self.__epsilon


def progress_bar(percent, epoch):
    """
    Prints an inline progress bar in the console with the epoch number.

    Args:
        percent (float): The completion percentage (between 0 and 100).
        epoch (int): The current epoch number.
    """

    bar_length = 50  # Length of the progress bar
    filled_length = int(bar_length * percent // 100)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f'\rEpoch {epoch} [{bar}] {percent:.2f}%')
    sys.stdout.flush()

def logarithmic(x, a, b):
    return a * np.log(x) + b

def exponential(x, a, b):
    return a * np.exp(-b * x)

# Main training loop
def main():
    # Define training parameters
    games_per_epoch = 100
    epochs = 2500

    # Initialize game environment
    agent_list = []
    mean_reward_list = []
    mean_loss_list = []
    loss = []
    n_step = 3
    gamma = 0.75
    max_bank = 10000
    # shared_memory = NStepReplayBuffer(100000000, n_step, gamma)
    shared_memory=None
    for i in range(number_of_agents):
        agent_list.append(Agent(state_size, action_size, n_step, gamma, shared_memory, max_bank))
    env = PokerGame(number_of_players=number_of_agents, ante=100, max_bank=max_bank, starting_stacks=1000)

    # Record the start time
    start_time = time.perf_counter()

    for epoch in range(epochs):

        games_won = np.zeros(number_of_agents)
        last_actions = [0] * number_of_agents  # Track the last agent each agent took
        last_state = np.zeros((number_of_agents, state_size))  # Track the last state that resulted in the last action
        final_state = [0] * state_size
        folded = [0] * number_of_agents
        for game in range(games_per_epoch):

            # starting_player = random.randint(0, 5)
            starting_player = 0
            while True:

                for i in range(number_of_agents):

                    # Determine which agent is taking an action
                    index = (starting_player + i) % number_of_agents
                    agent = agent_list[index]

                    # If the game is finished, no need to skim through the rest of the players
                    if env.is_game_finished():
                        break

                    if not env.is_player_playing(index):
                        continue

                    # Get the game state
                    state = env.encode_game_state()
                    # Get the state vector from the game state
                    state_vector = agent.get_state_vector(state, index)
                    # Agent takes action
                    agent_action = agent.act(state_vector)
                    decoded_action = agent.decode_action(agent_action)
                    # Map action index to actual action in the environment
                    executed_action = env.execute_agent_action(decoded_action)
                    # Update the action the agent think they took with the one they actually took
                    agent_action = agent.encode_action(executed_action)
                    # Keep track of how many times each agent folded
                    if executed_action == "F":
                        folded[index] = folded[index] + 1
                    # Update state vector based on the new state from the agent's action
                    next_state_vector = agent.get_state_vector(env.encode_game_state(), index)
                    # Get the reward from the environment based on the previous action
                    reward = env.get_reward(i)  # Environment tracks players based on I value, not index
                    # Record this reward for metrics
                    agent.update_epoch_reward(reward)
                    # Check if the game is finished
                    if env.is_game_finished():
                        done = 1
                        final_state = next_state_vector
                    else:
                        done = 0
                    # Remember the experience
                    agent.remember(state_vector, agent_action, reward, next_state_vector, done)
                    # Learn from experience
                    current_loss = agent.replay()
                    if current_loss is not None:
                        loss.append(current_loss)
                    # Store this action as the last action the agent took
                    last_actions[index] = agent_action
                    last_state[index] = state_vector

                # Break out of the loop if the game is over
                if env.is_game_finished():
                    break

            # Clean up the game before starting the next
            winner, amount_won = env.get_game_winner()
            games_won[winner] = games_won[winner] + 1
            # print("Epoch " + str(epoch) + ", Game " + str(game) + ": Winner was Agent " + str(winner) + " with a $" + str(amount_won) + " pot")
            progress_bar((game / games_per_epoch) * 100, epoch)

            # Assign final reward value for each agent
            end_rewards = env.finish_game()
            for i in range(number_of_agents):
                # Get the correct player assignments
                index = (starting_player + i) % number_of_agents
                agent = agent_list[index]

                # Get the final state and reward
                reward = end_rewards[index]

                # Store the final experience
                agent.remember(last_state[index], last_actions[index], reward, final_state, 1)

                # Learn from experience
                current_loss = agent.replay()
                if current_loss is not None:
                    loss.append(current_loss)

        # Print the metrics for each agent
        progress_bar(100, epoch)  # Finish the progress bar
        rewards = []
        print("")
        print("Epoch " + str(epoch) + " results:")
        for idx, agent in enumerate(agent_list):
            print(
                f"Agent {idx}, Games Won: {games_won[idx]}, Games folded: {folded[idx]}, Agent bank: ${env.get_player_bank(idx)}, Epsilon: {agent.get_epsilon()}, Total Reward: {agent.get_epoch_reward()}")
            rewards.append(agent.get_epoch_reward())
            agent.reset_epoch_reward()
        print("Average reward for this Epoch " + str(epoch) + ": " + str(np.mean(rewards)))
        mean_reward_list.append(np.mean(rewards))
        if not None in loss and len(loss) > 0:
            mean_loss_list.append(np.mean([tensor.item() for tensor in loss]))
        loss = []

        # Save the weights every 100 epochs
        if epoch % 10 == 0:
            # Save the trained model that had the highest number of games won
            best_agent_idx = np.where(rewards == np.max(rewards))[0][0]
            print("Agent " + str(
                best_agent_idx) + " had the best performance in the last epoch. Saving the model weights.")
            torch.save(agent_list[best_agent_idx].get_policy_network().state_dict(), 'dqn_poker_model.pth')

    # Record the end time
    end_time = time.perf_counter()

    # Calculate the execution time
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")

    # Calculate regression of the mean rewards
    x = np.linspace(1, len(mean_reward_list), len(mean_reward_list))
    param, param_cov = curve_fit(logarithmic, x, mean_reward_list)

    # Plot mean rewards over time
    plt.plot(x, mean_reward_list, 'yo', x, logarithmic(x, param[0], param[1]), '--k')
    plt.ylabel('Mean Reward Value')
    plt.xlabel('Epoch')
    plt.title('Mean Reward Value per Epoch')
    plt.show()

    # Calculate regression of the mean loss. This data looks more exponential in nature, so we map the linear to an exponential curve
    mean_loss_list.pop(0) # Ignore the first value since it skews the data too hard
    x = np.linspace(1, len(mean_loss_list), len(mean_loss_list))
    param, param_cov = curve_fit(exponential, x, mean_loss_list)

    # Plot mean loss over time
    plt.plot(x, mean_loss_list, 'yo', x, exponential(x, param[0], param[1]), '--k')
    plt.ylabel('Mean Loss Value')
    plt.xlabel('Epoch')
    plt.title('Mean Loss Value per Epoch')
    plt.show()

    # Save the trained model that had the highest number of games won
    best_agent_idx = np.where(rewards == np.max(rewards))[0][0]
    print("Agent " + str(best_agent_idx) + " had the best performance in the last epoch. Saving the model weights.")
    torch.save(agent_list[best_agent_idx].get_policy_network().state_dict(), 'dqn_poker_model.pth')


if __name__ == "__main__":
    main()
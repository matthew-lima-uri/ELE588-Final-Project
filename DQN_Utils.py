import numpy as np
from collections import deque
import random
from copy import deepcopy

number_of_agents = 5
state_size = (2 * 2) + 1 + (5 * 2) + 1 + 1 + ((number_of_agents - 1) * 1) + ((number_of_agents - 1) * 1) + (
            (number_of_agents - 1) * 1)
action_size = 4

# Function to map PokerKit card definitions to Agent definitions
def get_card(card):
    rank_dict = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10,
                 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    suit_dict = {'h': 1, 'd': 2, 'c': 3, 's': 4}

    return rank_dict[card[0]], suit_dict[card[1]]

# Function to convert the state from the environment into a neural network input
def get_state_vector(state, player_index, max_pot):

    # Initialize the state vector
    state_vector = np.zeros(state_size)

    # Encode private cards
    if not (state.player_cards[player_index] is None):
        state_vector[0: 2] = get_card(state.player_cards[player_index][0])
        state_vector[2: 4] = get_card(state.player_cards[player_index][1])
        state_vector[0] = state_vector[0] / 14  # Max value for a rank
        state_vector[1] = state_vector[1] / 4  # Max value for a suit
        state_vector[2] = state_vector[0] / 14  # Max value for a rank
        state_vector[3] = state_vector[1] / 4  # Max value for a suit

    # Encode own bank
    state_vector[4] = state.player_banks[player_index] / max_pot  # Max value for a bank

    # Encode community cards
    for i in range(len(state.board_cards)):
        if not (state.board_cards[i] is None):
            state_vector[5 + (i * 2): 5 + (i * 2) + 2] = get_card(state.board_cards[i])
            state_vector[5 + (i * 2)] = state_vector[5 + (i * 2)] / 14  # Max value for a rank
            state_vector[5 + (i * 2) + 1] = state_vector[5 + (i * 2) + 1] / 4  # Max value for a suit
            # Community cards that have an unknown value will remain 0

    # Encode the pool
    state_vector[15] = state.pot / (max_pot * number_of_agents)  # Maximum theoretical pot

    # Encode the betting round
    state_vector[16] = state.betting_round / 4  # Max value for betting rounds

    # Encode the other player statuses
    agent_index = 0
    vector_index = 17
    for i in range(number_of_agents):
        # Skip the own players status
        if i == player_index:
            continue
        state_vector[vector_index] = state.player_statuses[agent_index]  # Already capped between 0 and 1
        agent_index = agent_index + 1
        vector_index = vector_index + 1

    # Encode the other player bets
    agent_index = 0
    for i in range(number_of_agents):
        # Skip the own players status
        if i == player_index:
            continue
        state_vector[vector_index] = state.player_bets[
                                             agent_index] / max_pot  # Theoretical max bet per player
        agent_index = agent_index + 1
        vector_index = vector_index + 1

    # Encode the other player banks
    agent_index = 0
    for i in range(number_of_agents):
        # Skip the own players bank
        if i == player_index:
            continue
        state_vector[vector_index] = state.player_banks[agent_index] / max_pot  # Max bank per player
        agent_index = agent_index + 1
        vector_index = vector_index + 1

    return state_vector

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
        self.capacity = capacity
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def push(self, experience):
        self.n_step_buffer.append(experience)
        if len(self.n_step_buffer) == self.n_step:
            # state, action, _, _, _ = self.n_step_buffer[0]
            # reward, next_state, done = self._get_n_step_info()
            # self.buffer.append((state, action, reward, next_state, done))
            self.buffer.append(self.n_step_buffer[0])

    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][2:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            _, _, r, ns, d = transition
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (ns, d) if d else (next_state, done)
        return reward, next_state, done

    def sample(self, batch_size):
        batch_index = random.randint(0, len(self.buffer) - batch_size) # Need to be at least batch size away
        return list(self.buffer)[batch_index : batch_index + batch_size]

    def get_buffers(self):
        return self.buffer, self.n_step_buffer

    def copy(self, memory_buffer, n_step_buffer):
        self.buffer = deepcopy(memory_buffer)
        self.n_step_buffer = deepcopy(n_step_buffer)

    def __len__(self):
        return len(self.buffer)
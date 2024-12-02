
import numpy as np
import torch
import signal
import DQN_Utils
from ProbabilityAgent import ProbabilityAgent
from gym import PokerGame
import time
import sys
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from DQNAgent import DoubleDQNAgent, DuelingDQNAgent

"""
System constants

State Size: Defines the input space for the model. Derived from own cards, own bank, community cards, current pot, 
betting round, other player statuses, other player bets, other player banks.

Action Size: Defines the output space for the model. Derived from the possible actions (Fold, Call, Raise, All-in)
"""
number_of_agents = 5
state_size = (2 * 2) + 1 + (5 * 2) + 1 + 1 + ((number_of_agents - 1) * 1) + ((number_of_agents - 1) * 1) + (
            (number_of_agents - 1) * 1)
action_size = 4
continue_running = True

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
    epochs = 10000

    # Initialize game environment
    agent_list = []
    mean_reward_list = []
    mean_loss_list = []
    n_step = 6
    gamma = 0.99
    max_bank = 10000
    shared_memory=None
    for i in range(int(number_of_agents) - 1):
        agent_list.append(DuelingDQNAgent(state_size, action_size, n_step=n_step, gamma=gamma, memory=shared_memory, max_pot=max_bank, learning_rate_step=games_per_epoch*10))
    agent_list.append(ProbabilityAgent(0.1, 0.4, 0.2, 0.3))
    env = PokerGame(number_of_players=number_of_agents, ante=100, max_bank=max_bank, starting_stacks=1000, minimum_bet=50)

    # Record the start time
    start_time = time.perf_counter()

    for epoch in range(epochs):

        games_won = np.zeros(number_of_agents)
        last_actions = [0] * number_of_agents  # Track the last agent each agent took
        last_state = np.zeros((number_of_agents, state_size))  # Track the last state that resulted in the last action
        final_state = [0] * state_size
        folded = [0] * number_of_agents
        loss = [0] * number_of_agents
        for game in range(games_per_epoch):

            betting_round = 1
            while True:

                for index, agent in enumerate(agent_list):

                    # If the game is finished, no need to skim through the rest of the players
                    if env.is_game_finished():
                        break

                    if not env.is_player_playing(index):
                        continue

                                # Check if the current round of betting ended
                    if env.get_betting_round() > betting_round:
                        betting_round = env.get_betting_round()
                        break

                    # If the player went all in, skip their turn
                    if env.get_player_bank(index) == 0:
                        continue


                    # Get the game state
                    state = env.encode_game_state()
                    # Get the state vector from the game state
                    state_vector = DQN_Utils.get_state_vector(state, index, max_bank)
                    # Agent takes action
                    agent_action = agent.act(state_vector)
                    decoded_action = DQN_Utils.decode_action(agent_action)
                    # Map action index to actual action in the environment
                    executed_action = env.execute_agent_action(index, decoded_action)
                    # Update the action the agent think they took with the one they actually took
                    agent_action = DQN_Utils.encode_action(executed_action)
                    # Keep track of how many times each agent folded
                    if executed_action == "F" and not env.is_player_playing(index):
                        folded[index] = folded[index] + 1
                    elif executed_action == "F" :
                        # Something went wrong and they didn't fold when they thought they did
                        raise Exception("Agent action failure!")
                    # Update state vector based on the new state from the agent's action
                    next_state_vector = DQN_Utils.get_state_vector(env.encode_game_state(), index, max_bank)
                    # Get the reward from the environment based on the previous action
                    reward = env.get_reward(index)  # Environment tracks players based on I value, not index
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
                        loss[index] = loss[index] + current_loss
                    # Store this action as the last action the agent took
                    last_actions[index] = agent_action
                    last_state[index] = state_vector

                # Break out of the loop if the game is over
                if env.is_game_finished():
                    break

            # Clean up the game before starting the next
            winners = env.get_game_winners()
            for winner in winners:
                games_won[winner] = games_won[winner] + 1
            # print("Epoch " + str(epoch) + ", Game " + str(game) + ": Winner was Agent " + str(winner) + " with a $" + str(amount_won) + " pot")
            progress_bar((game / games_per_epoch) * 100, epoch)

            # Assign final reward value for each agent
            end_rewards = env.finish_game()
            for index, agent in enumerate(agent_list):

                # Get the final state and reward
                reward = end_rewards[index]
                agent.update_epoch_reward(reward)

                # Store the final experience
                agent.remember(last_state[index], last_actions[index], reward, final_state, 1)

                # Learn from experience
                current_loss = agent.replay()
                if current_loss is not None:
                    loss[index] = loss[index] + current_loss

        # Print the metrics for each agent
        progress_bar(100, epoch)  # Finish the progress bar
        rewards = []
        print("")
        print("Epoch " + str(epoch) + " results:")
        for idx, agent in enumerate(agent_list):
            print(
                f"Agent {idx}, Games Won: {games_won[idx]}, Games folded: {folded[idx]}, Agent bank: ${env.get_player_bank(idx)}, Epsilon: {agent.get_epsilon()}, Total Reward: {agent.get_epoch_reward()}, Total Loss: {loss[idx]}")
            if type(agent) == DuelingDQNAgent or type(agent) == DoubleDQNAgent:
                rewards.append(agent.get_epoch_reward())
            agent.reset_epoch_reward()
        print("Average reward for this Epoch " + str(epoch) + ": " + str(np.mean(rewards)))
        mean_reward_list.append(np.mean(rewards))
        if not None in loss and len(loss) > 0:
            mean_loss = np.mean(loss)
            print("Average loss for this Epoch " + str(epoch) + ": " + str(mean_loss))
            mean_loss_list.append(mean_loss)

        # Save the weights every 100 epochs
        if epoch % 10 == 0:
            # Save the trained model that had the highest number of games won
            best_agent_idx = np.where(rewards == np.max(rewards))[0][0]
            if type(agent_list[best_agent_idx]) is not ProbabilityAgent:
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
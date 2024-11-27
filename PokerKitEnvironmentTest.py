from gym import PokerGame
from TexasAgent import Agent
import os

num_players = 6
real_players = 1
env = PokerGame(number_of_players=num_players, ante=100, minimum_bet=1, starting_stacks=1000, max_bank=10000)
agents = []
for i in range(num_players - real_players):
    agents.append(Agent(32, 4, ))
    agents[-1].load_policy_network('dqn_poker_model.pth')


# Helper function to get the players action as string input.
def get_player_action():
    action = input("Enter action: Fold(F), Call/Check(C), Raise(R), All-In (A): ")
    amount = 0
    if action.upper() == "R":
        amount = input("Enter raise amount: ")
        amount = int(amount)
    return action, amount


# Determine if debug mode is on
debug_mode_str = input("Debug mode? (Y/N)")
debug_mode = False
if debug_mode_str.upper() == "Y":
    debug_mode = True

# Loop until the players wish to quit
while True:

    # Loop until the game ends
    betting_round = 1
    while not env.is_game_finished():

        # Loop through player actions
        for player in range(num_players):

            # Clear the screen
            if not debug_mode:
                os.system('cls')

            # Check if the game ended as a result of the player actions
            if env.is_game_finished():
                break

            # Check if the player folded
            if not env.is_player_playing(player):
                continue

            # Check if the current round of betting ended
            if env.get_betting_round() > betting_round:
                betting_round = env.get_betting_round()
                break

            # Process real player data
            if player < real_players:

                show_information = input("Show information for Player " + str(player + 1) + "? (Y/N)")
                if show_information.upper() == "Y":
                    print("")
                    print("Operations this round of betting: ")
                    print(env.get_relevant_operations_this_round())

                    print("Player states:")
                    for f in range(num_players):
                        if not env.is_player_playing(f):
                            print("Player " + str(f + 1) + " is folded")
                        else:
                            print("Player " + str(f + 1) + " has $" + str(env.get_player_bank(f)) + " in their bank")

                    print("")
                    print("Player " + str(player + 1) + " information:")
                    player_hand = env.get_player_cards(player)
                    board_hand = env.get_board_cards()
                    print("Hole cards: ")
                    [print(card.__str__()) for card in player_hand]
                    print("Board cards: ")
                    [print(card.__str__()) for card in board_hand]
                    print("Best hand: " + str(env.get_best_hand(player)))

                    print("Current betting round: " + str(env.get_betting_round()))
                    print("Current call: $" + str(env.get_current_bet(player)))
                    print("Current pot: $" + str(env.get_current_pot()))
                    print("Current bank: $" + str(env.get_player_bank(player)))

                while True:
                    try:
                        a, r = get_player_action()
                        env.execute_agent_action(a, r)
                        break
                    except Exception as e:
                        print("Failed to execute action because:")
                        print(e.__str__())
                        print("Try a new action...")
                print("")

            else:

                if debug_mode:
                    print("Player " + str(player + 1) + " information:")
                    player_hand = env.get_player_cards(player)
                    board_hand = env.get_board_cards()
                    print("Hole cards: ")
                    [print(card.__str__()) for card in player_hand]
                    print("Player + " + str(player + 1) + " Best hand: " + str(env.get_best_hand(player)))
                    print("")

                game_state = agents[player - real_players].get_state_vector(env.encode_game_state(), player)
                action = agents[player - real_players].decode_action(
                    agents[player - real_players].act(game_state, True))
                env.execute_agent_action(action)

    if debug_mode:
        analyze_data = input("Press enter to make next action...")

    # End game information
    winning_player = env.get_winning_player()
    print("The winner is: Player " + str(winning_player + 1))
    print("The winning hand was:")
    print(str(env.get_best_hand(winning_player)))
    print("The final pot is: $" + str(env.get_final_pot()))

    play_again = input("Do you want to play another game? (Y/N):")
    if play_again.upper() == "Y":
        env.finish_game()
    else:
        break
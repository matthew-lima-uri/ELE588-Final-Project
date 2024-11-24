from gym import PokerGame
from TexasAgent import Agent

num_players = 6
real_players = 1
env = PokerGame(number_of_players=num_players, ante=1, minimum_bet=1, starting_stacks=100)
agents = []
for i in range(num_players - real_players):
    agents.append(Agent(32, 4, ))
    agents[-1].load_policy_network('dqn_poker_model.pth')

# Helper function to get the players action as string input.
def get_player_action():
    action = input("Enter action: Fold(F), Call/Check(C), Raise(R), All-In (A): ")
    amount = 0
    if action == "R":
        amount = input("Enter raise amount: ")
        amount = int(amount)
    return action, amount

# Loop until the players wish to quit
while True:

    # Loop until the game ends
    while True:

        # Loop through player actions
        for player in range(real_players):
            if not env.is_player_playing(player):
                continue

            # If the game ended, skip the rest of the player turns
            if env.is_game_finished():
                break
            show_information = input("Show information for Player " + str(player + 1) + "? (Y/N)")
            if show_information.upper() == "Y":
                print("Player " + str(player + 1) + " information:")
                player_hand = env.get_player_cards(player)
                board_hand = env.get_board_cards()
                print("Hole cards: ")
                [print(card.__str__()) for card in player_hand]
                print("Board cards: ")
                [print(card.__str__()) for card in board_hand]
                print("Best hand: " + str(env.get_best_hand(player)))

                print("Current betting round: " + str(env.get_betting_round()))
                print("Current bet: $" + str(env.get_current_bet(player)))
                print("Current pot: $" + str(env.get_current_pot()))
                print("Current bank: $" + str(env.get_player_bank(player)))

            while True:
                try:
                    a,r = get_player_action()
                    env.execute_agent_action(a, r)
                    break
                except Exception as e:
                    print("Failed to execute action because:")
                    print(e.__str__())
                    print("Try a new action...")
            print("")

        # Check if the game ended as a result of the player actions
        if env.is_game_finished():
            break

        # Loop through the AI actions
        for ai in range(num_players - real_players):
            if env.is_player_playing(ai) and not env.is_game_finished():
                game_state = agents[ai].get_state_vector(env.encode_game_state(), ai)
                action = agents[ai].decode_action(agents[ai].act(game_state, True))
                env.execute_agent_action(action)

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
from gym import PokerGame

num_players = 6
real_players = 2
game = PokerGame(number_of_players=num_players, ante=1, minimum_bet=1, starting_stacks=100)

# Helper function to get the players action as string input.
def get_player_action():
    action = input("Enter action: Fold(F), Call/Check(C), Raise(R): ")
    amount = 0
    if action == "R":
        amount = input("Enter raise amount: ")
        amount = int(amount)
    return action, amount


# Loop until the game ends
while True:

    # Loop through player actions
    for player in range(real_players):
        if not game.is_player_playing(player):
            continue

        # If the game ended, skip the rest of the player turns
        if game.is_game_finished():
            break
        show_information = input("Show information for Player " + str(player + 1) + "? (Y/N)")
        if show_information.upper() == "Y":
            print("Player " + str(player + 1) + " information:")
            player_hand = game.get_player_cards(player)
            board_hand = game.get_board_cards()
            print("Hole cards: ")
            [print(card.__str__()) for card in player_hand]
            print("Board cards: ")
            [print(card.__str__()) for card in board_hand]
            print("Best hand: " + str(game.get_best_hand(player)))

            print("Current bet: $" + str(game.get_current_bet()))
            print("Current pot: $" + str(game.get_current_pot()))
            print("Current bank: $" + str(game.get_player_bank(player)))

        while True:
            try:
                a,r = get_player_action()
                game.execute_agent_action(a,r)
                break
            except Exception as e:
                print("Failed to execute action because:")
                print(e.__str__())
                print("Try a new action...")
        print("")

    # Check if the game ended as a result of the player actions
    if game.is_game_finished():
        break

    # Loop through the AI actions
    for ai in range(real_players, num_players):
        game.execute_agent_action("C", 0)

# End game information
winning_player = game.get_winning_player()
print("The winner is: Player " + str(winning_player + 1))
print("The winning hand was:")
print(str(game.get_best_hand(winning_player)))
print("The final pot is: $" + str(game.get_final_pot()))
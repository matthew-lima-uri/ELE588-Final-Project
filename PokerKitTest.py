"""
Example on how to use the PokerKit library.
Creates a simulated single game of Hold'Em with 5 other players.
"""
from pokerkit import Automation, NoLimitTexasHoldem, Mode
import sys

num_players=6
game = NoLimitTexasHoldem.create_state(
    # Automations
    (
        Automation.ANTE_POSTING,
        Automation.BET_COLLECTION,
        Automation.BLIND_OR_STRADDLE_POSTING,
        Automation.CARD_BURNING,
        Automation.BOARD_DEALING,
        Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
        Automation.HAND_KILLING,
        Automation.CHIPS_PUSHING,
        Automation.CHIPS_PULLING,
        Automation.RUNOUT_COUNT_SELECTION
    ),
    ante_trimming_status=True,  # Uniform antes?
    raw_antes=1,  # Antes
    min_bet=1,  # Min-bet
    raw_blinds_or_straddles=(0,0,0,0,0,0),
    raw_starting_stacks=(5,5,5,5,5,5),
    player_count=num_players,  # Number of players
    mode=Mode.CASH_GAME
)

# Helper function to get the players action
def get_player_action():
    action = input("Enter action: Fold(F), Call/Check(C), Raise(R): ")
    if action == "F":
        if game.can_fold():
            game.fold()
            print("You have folded. The game is over.")
            ys.exit(0)
        else:
            print("Screw you; you can't fold!")
            game.check_or_call()
    elif action == "C":
        game.check_or_call()
    elif action == "R":
        amount = input("Enter raise amount: ")
        amount = int(amount)
        if amount > game.get_effective_stack(0):
            raise Exception("You ain't got that much money")
        if amount == game.get_effective_stack(0):
            game.complete_bet_or_raise_to(amount)
            # Get other players to bet
            for i in range(num_players - 1):
                # All other players will call
                game.check_or_call()
            end_game()
        else:
            game.complete_bet_or_raise_to(amount)
    else:
        raise Exception("Not a valid input")

def display_hand(hand):
    for h in range(len(hand)):
        print("Card " + str(h) + ": " + hand[h].__str__())

def end_game():
    # Display the winning hand, and determine if it was the player who won
    if game.all_in_status:
        print("All in!")
    if game.status:
        raise Exception("The game is not over...")
    else:
        winning_player = game.operations[-1].player_index
        print("The winner is: Player " + str(winning_player))
        print("The winning hand was:")
        print(str(game.get_hand(winning_player,0,0)))
        print("Your hand was: " + str(game.get_hand(0,0,0)))
        print("The final pot is: $" + str(game.operations[-1].amount))
        sys.exit(0)



# Deal hole cards
for i in range(num_players*2):
    game.deal_hole(player_index=i%num_players)
player_cards = tuple(game.get_down_cards(0))
display_hand(player_cards)

# First round of betting
get_player_action()
for i in range(num_players - 1):
    # All other players will call
    game.check_or_call()
print("Current pot is: $" + str(game.total_pot_amount))
print("")

# Deal the flop
board_cards = tuple(game.get_board_cards(0))
print("Board cards:")
display_hand(board_cards)
print("Best hand: " + str(game.get_hand(0,0,0)))

# Second round of betting
get_player_action()
for i in range(num_players - 1):
    # All other players will call
    game.check_or_call()
print("Current pot is: $" + str(game.total_pot_amount))
print("")

# Deal the turn
board_cards = tuple(game.get_board_cards(0))
print("Board cards:")
display_hand(board_cards)
print("Best hand: " + str(game.get_hand(0,0,0)))

# Third round of betting
get_player_action()
for i in range(num_players - 1):
    # All other players will call
    game.check_or_call()
print("Current pot is: $" + str(game.total_pot_amount))
print("")

# Deal the river
board_cards = tuple(game.get_board_cards(0))
print("Board cards:")
display_hand(board_cards)
print("Best hand: " + str(game.get_hand(0,0,0)))

# Final round of betting
get_player_action()
for i in range(num_players - 1):
    # All other players will call
    game.check_or_call()
print("Current pot is: $" + str(game.total_pot_amount))
print("")

end_game()
import numpy as np
import pokerkit
from pokerkit import NoLimitTexasHoldem, Automation, Mode

"""
Helpful PokerKit state properties
turn_index = Index number of whoever the game is waiting on to take an action

"""


class GameState:

    def __init__(self, player_cards, board_cards, pot, betting_round, player_bets, player_banks, player_statuses,
                 all_in):
        self.player_cards = player_cards
        self.board_cards = board_cards
        self.pot = pot
        self.betting_round = betting_round
        self.player_bets = player_bets
        self.player_banks = player_banks
        self.player_statuses = player_statuses
        self.all_in = all_in


class PokerGame:

    def __init__(self, number_of_players=6, ante=1, minimum_bet=1, starting_stacks=100, max_bank=1000):

        self.__number_of_players = number_of_players
        self.__game = None
        self.__ante = ante
        self.__min_bet = minimum_bet
        self.__starting_stack = starting_stacks
        self.__last_raise = 0
        self.__last_call = 0
        self.reset((starting_stacks,) * number_of_players)
        self.__max_bank = max_bank
        self.__max_possible_bet = max_bank * number_of_players

    def reset(self, starting_stacks):

        self.__game = NoLimitTexasHoldem.create_state(
            # Automations
            (
                Automation.ANTE_POSTING,
                Automation.BET_COLLECTION,
                Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.CARD_BURNING,
                Automation.HOLE_DEALING,
                Automation.BOARD_DEALING,
                Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING,
                Automation.CHIPS_PULLING,
                Automation.RUNOUT_COUNT_SELECTION
            ),
            ante_trimming_status=True,  # Uniform antes?
            raw_antes=self.__ante,  # Antes
            min_bet=self.__min_bet,  # Min-bet
            raw_blinds_or_straddles=(0,) * self.__number_of_players,
            raw_starting_stacks=starting_stacks,
            player_count=self.__number_of_players,  # Number of players
            mode=Mode.CASH_GAME
        )

    def finish_game(self):

        # Determine if any of the bots lost all their money, and if so give them a stimulus check
        current_stacks = self.__game.stacks
        rewards = np.zeros(len(current_stacks))
        for idx, bank in enumerate(current_stacks):
            if bank == 0:
                current_stacks[idx] = self.__starting_stack
                rewards[idx] = -1
            # If this player has the max amount of money allowed
            elif bank >= self.__max_bank:
                current_stacks[idx] = self.__max_bank
                rewards[idx] = 1
            else:
                rewards[idx] = current_stacks[idx] / self.__max_bank
        self.reset(current_stacks)
        return rewards

    def encode_game_state(self):

        # Encode the player cards
        encoded_player_cards = []
        encoded_board_cards = []

        if not self.is_game_finished():
            for i in range(self.__number_of_players):
                if self.is_player_playing(i):
                    card_values = self.get_player_cards(i)
                    card_values_str = [card_values[0].__str__()[-3:-1], card_values[1].__str__()[-3:-1]]
                else:
                    card_values_str = None
                encoded_player_cards.append(card_values_str)

            # Encode the board cards
            board_cards = self.get_board_cards()
            for i in range(len(board_cards)):
                encoded_board_cards.append(board_cards[i].__str__()[-3:-1])
        else:
            for i in range(self.__number_of_players):
                encoded_player_cards.append(None)
            for i in range(len(self.get_board_cards())):
                encoded_board_cards.append(None)

        # Encode the pot
        pot = self.get_current_pot()

        # Encode the betting round
        betting_round = self.get_betting_round()

        # Encode the player bets
        player_bets = self.__game.bets

        # Encode the player banks
        banks = []
        for i in range(self.__number_of_players):
            banks.append(self.get_player_bank(i))

        # Encode the player statuses
        player_statuses = self.__game.statuses

        # Encode all in status
        all_in = self.__game.all_in_status

        return GameState(player_cards=encoded_player_cards, board_cards=encoded_board_cards, pot=pot,
                         betting_round=betting_round, player_bets=player_bets, player_banks=banks,
                         player_statuses=player_statuses, all_in=all_in)

    def get_betting_round(self):
        round = len(self.__game.burn_cards)
        if round is None:
            round = 0
        return round + 1

    def get_player_cards(self, player_index):
        return list(self.__game.get_down_cards(player_index))

    def get_board_cards(self):
        return list(self.__game.get_board_cards(0))

    def get_current_bet(self, player_index):
        return self.__game.checking_or_calling_amount

    def get_current_pot(self):
        return self.__game.total_pot_amount

    def get_best_hand(self, player_index):
        return self.__game.get_hand(player_index, 0, 0)

    def get_player_bank(self, player_index):
        return self.__game.stacks[player_index]

    def get_winning_player(self):
        return self.__game.operations[-1].player_index

    def get_final_pot(self):
        return self.__game.operations[-1].amount

    def is_player_playing(self, player_index):
        return self.__game.statuses[player_index]

    def is_game_finished(self):
        return not self.__game.status

    def get_game_winner(self):
        return self.__game.operations[-1].player_index, self.__game.operations[-1].amount

    def get_relevant_operations_this_round(self):

        # Get all operations between this betting round and the previous
        operation_log = []
        for operation in reversed(self.__game.operations):
            if type(operation) != pokerkit.CardBurning:
                if type(operation) == pokerkit.CompletionBettingOrRaisingTo:
                    operation_log.append(
                        "Player " + str(operation.player_index + 1) + " raised $" + str(operation.amount) + "\n")
                elif type(operation) == pokerkit.CheckingOrCalling:
                    operation_log.append(
                        "Player " + str(operation.player_index + 1) + " called $" + str(operation.amount) + "\n")
                elif type(operation) == pokerkit.Folding:
                    operation_log.append("Player " + str(operation.player_index + 1) + " folded" + "\n")
            else:
                break

                # Now reverse it
        reversed_string_list = ""
        for op in reversed(operation_log):
            reversed_string_list = reversed_string_list + op
        return reversed_string_list

    def execute_agent_action(self, action, raise_amount=5):
        action = action.upper()
        # Determine the minimum and maximum betting amounts
        min_raise_amount = self.__game.min_completion_betting_or_raising_to_amount
        # If someone goes all-in, there is no longer a min raise amount
        if min_raise_amount is None:
            min_raise_amount = self.__game.completion_betting_or_raising_amount

        if action == "F":
            if self.__game.can_fold():
                self.__game.fold()
            else:
                # If folding isn't possible, check/call. This usually happens because the player is the first bet
                if self.__game.can_check_or_call():
                    self.__game.check_or_call()
                else:
                    raise Exception("Game error. Can't check or call.")
        elif action == "C":
            if self.__game.can_check_or_call():
                self.__game.check_or_call()
            else:
                # Can't check or call probably means they need to fold
                if self.__game.can_fold():
                    self.__game.fold()
                else:
                    raise Exception("Game error. Can't fold.")
        elif action == "R":
            # If the raise amount is less than the minimum, set it to the minimum
            if raise_amount < min_raise_amount:
                raise_amount = min_raise_amount
            # Ensure the player has enough to raise
            if raise_amount > self.__game.get_effective_stack(self.__game.turn_index):
                # Can't raise, so check or call
                if self.__game.can_check_or_call():
                    self.__game.check_or_call()
                    self.__last_call = self.__game.completion_betting_or_raising_amount
                else:
                    # Can't check or call probably means they need to fold
                    if self.__game.can_fold():
                        self.__game.fold()
                    else:
                        raise Exception("Game error. Can't fold.")
            else:
                try:
                    self.__game.complete_bet_or_raise_to(raise_amount)
                    self.__last_raise = raise_amount
                except:
                    self.__game.check_or_call()
        elif action == "A":
            # Set the raise amount to be the players entire bank
            raise_amount = self.get_player_bank(self.__game.turn_index)
            # Rest of this code is the same as raising...
            # If the raise amount is less than the minimum, set it to the minimum
            if raise_amount < min_raise_amount:
                raise_amount = min_raise_amount
            # Ensure the player has enough to raise
            if raise_amount > self.__game.get_effective_stack(self.__game.turn_index):
                # Can't raise, so check or call
                if self.__game.can_check_or_call():
                    self.__game.check_or_call()
                    self.__last_call = self.__game.completion_betting_or_raising_amount
                else:
                    # Can't check or call probably means they need to fold
                    if self.__game.can_fold():
                        self.__game.fold()
                    else:
                        raise Exception("Game error. Can't fold.")
            else:
                try:
                    self.__game.complete_bet_or_raise_to(raise_amount)
                    self.__last_raise = raise_amount
                except:
                    self.__game.check_or_call()
        else:
            raise Exception("Not a valid input")

        return self.encode_game_state()

    def get_reward(self, player_index):

        # Determine events that occurred between the previous state and the current state
        last_operation = None
        for operation in reversed(self.__game.operations):
            try:
                if operation.player_index == player_index:
                    last_operation = operation
                    break
            except AttributeError:
                continue

        # If the player hasn't performed yet, no reward
        if last_operation is None:
            return 0

        if type(last_operation) == pokerkit.CompletionBettingOrRaisingTo:
            # If the player went all in, heavily punish them
            if self.get_player_bank(player_index) == 0:
                return -8
            else:
                return -0.75 * (last_operation.amount / (self.get_player_bank(player_index) + last_operation.amount))
        elif type(last_operation) == pokerkit.CheckingOrCalling:
            # If there is money involved in the call, assign a slight negative reward
            if last_operation.amount > 0:
                return -0.5 * (last_operation.amount / (self.get_player_bank(player_index) + last_operation.amount))
            # If the call is a check, no negative reward needed
            else:
                return 0
        elif type(last_operation) == pokerkit.Folding:
            # Reward loss is the total amount of money the player invested into this game
            return -1 * (self.__game.bets[player_index] + self.__game.antes[player_index] / (
                        self.get_player_bank(player_index) + self.__game.bets[player_index] + self.__game.antes[
                    player_index]))
        elif type(last_operation) == pokerkit.ChipsPulling:
            return 1

        return 0
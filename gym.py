from pokerkit import NoLimitTexasHoldem, Automation, Mode

class PokerGame:

    def __init__(self, number_of_players=6, ante=1, minimum_bet=1, starting_stacks=100):

        self.__reward = 0
        self.__number_of_players = number_of_players
        self.__game = None
        self.__ante = ante
        self.__min_bet = minimum_bet
        self.__stacks = (starting_stacks,) * number_of_players
        self.reset()

    def reset(self):

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
            raw_starting_stacks=self.__stacks,
            player_count=self.__number_of_players,  # Number of players
            mode=Mode.CASH_GAME
        )

    def get_player_cards(self, player_index):
        return list(self.__game.get_down_cards(player_index))

    def get_board_cards(self):
        return list(self.__game.get_board_cards(0))

    def get_current_bet(self):
        return self.__game.checking_or_calling_amount

    def get_current_pot(self):
        return self.__game.total_pot_amount

    def get_best_hand(self, player_index):
        return self.__game.get_hand(player_index,0,0)

    def get_player_bank(self, player_index):
        return self.__game.get_effective_stack(player_index)

    # TODO: Make a better solution than this. This assumes the last operation is the chip pull
    def get_winning_player(self):
        return self.__game.operations[-1].player_index

    # TODO: Make a better solution than this. This assumes the last operation is the chip pull
    def get_final_pot(self):
        return self.__game.operations[-1].amount

    def is_player_playing(self, player_index):
        return self.__game.statuses[player_index]

    def is_game_finished(self):
        return not self.__game.status

    def execute_agent_action(self, action, raise_amount=0):
        action = action.upper()
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
                raise Exception("Game error. Can't check or call.")
        elif action == "R":
            raise_amount = raise_amount + self.get_current_bet()
            if raise_amount > self.__game.get_effective_stack(0):
                raise Exception("You ain't got that much money")
            else:
                self.__game.complete_bet_or_raise_to(raise_amount)
        else:
            raise Exception("Not a valid input")

    def get_reward(self):

        # Determine events that occurred between the previous state and the current state

        # Assign values to the differences (i.e. player winning, player folding, etc.)

        return self.__reward
from pokerkit import NoLimitTexasHoldem, Automation, Mode

class Environment:

    def __init__(self, number_of_players):

        self.__community_cards = ""
        self.__reward = 0
        self.__game = NoLimitTexasHoldem(
            automations=(Automation.ANTE_POSTING,
                         Automation.BET_COLLECTION,
                         Automation.BLIND_OR_STRADDLE_POSTING,
                         Automation.CARD_BURNING,
                         Automation.HOLE_DEALING,
                         Automation.BOARD_DEALING,
                         Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                         Automation.HAND_KILLING,
                         Automation.CHIPS_PUSHING,
                         Automation.CHIPS_PULLING),
        ante_trimming_status=True,
        raw_antes=1,
        raw_blinds_or_straddles=(2, 0),
        min_bet=1,
        mode=Mode.CASH_GAME)

    def reset(self):

        self.__community_cards = ""

        return self.get_game_state()

    def get_game_state(self):

        return self.__game

    def execute_agent_action(self, action):

        return self.get_game_state()

    def get_reward(self):

        # Determine events that occurred between the previous state and the current state

        # Assign values to the differences (i.e. player winning, player folding, etc.)

        return self.__reward
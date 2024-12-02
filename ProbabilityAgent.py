import random
from Agent import Agent


class ProbabilityAgent(Agent):

    def __init__(self, all_in_p, fold_p, raise_p, call_p):

        self.__all_in_p = all_in_p
        self.__fold_p = fold_p
        self.__raise_p = raise_p
        self.__call_p = call_p

        if all_in_p + fold_p + raise_p + call_p != 1:
            raise Exception("Probability values must add to 1!")

        self.__weights = [self.__all_in_p, self.__fold_p, self.__raise_p, self.__call_p]
        self.__actions = [0, 1, 2, 3]
        self.__epoch_reward = 0

    def act(self, state):

        return random.choices(self.__actions, self.__weights, k=1)[0]

    def update_epoch_reward(self, new_reward):
        self.__epoch_reward = self.__epoch_reward + new_reward

    def get_epoch_reward(self):
        return self.__epoch_reward

    def reset_epoch_reward(self):
        self.__epoch_reward = 0

    def remember(self, state_vector, agent_action, reward, next_state_vector, done):
        pass

    def replay(self):
        return None

    def get_epsilon(self):
        return 1
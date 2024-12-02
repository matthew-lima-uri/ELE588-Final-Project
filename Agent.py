from abc import abstractmethod


class Agent():

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def update_epoch_reward(self, new_reward):
        pass

    @abstractmethod
    def get_epoch_reward(self):
        pass

    @abstractmethod
    def reset_epoch_reward(self):
        pass

    @abstractmethod
    def remember(self, state_vector, agent_action, reward, next_state_vector, done):
        pass

    @abstractmethod
    def replay(self):
        pass

    @abstractmethod
    def get_epsilon(self):
        pass
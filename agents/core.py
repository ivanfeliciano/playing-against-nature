import numpy as np

class Agent(object):
    def __init__(self, nature):
        self.nature = nature
        self.rewards_per_round = [0]
        self.n_rounds = 0
    def training(self, rounds):
        raise NotImplementedError
    def make_decision(self):
        raise NotImplementedError
    def get_rewards(self):
        return self.rewards_per_round
import random

import gym
import numpy as np

from env.light_env import LightEnv
from causal_structure import CausalStructure
from utils.lights_env_helper import powerset

class Environment(object):
    def __init__(self, env, adj_list, stochastic=False, true_action_prob=0.8):
        self.stochastic = stochastic
        self.causal_structure = CausalStructure(adj_list)
        self.env = env
        self.true_action_prob = true_action_prob
    def reset(self):
        raise NotImplementedError
    def step(self, action):
        raise NotImplementedError
    def init_q_table(self):
        raise NotImplementedError
    def sample_action(self):
        return self.env.action_space.sample()

class LightAndSwitchEnv(Environment):
    def __init__(self, env, adj_list, stochastic=False):
        self.env = env
        self.num = self.env.num
        self.horizon = self.num
        self.n_actions = self.num + 1
        super().__init__(env, adj_list, stochastic)
    def init_q_table(self):
        all_states = powerset(self.num)
        self.number_of_states = len(all_states)
        return {
            state :  np.zeros(self.n_actions) for state in all_states
        }
    def map_obs(self, state):
        return tuple(state[:self.num].astype(int))
    def reset(self):
        return self.map_obs(self.env.reset())
    def get_goal(self):
        return self.env.goal
    def get_state(self):
        return self.env._get_obs()[:self.num]
    def step(self, action):
        if self.stochastic and np.random.uniform() > self.true_action_prob:
            remain_actions = [i for i in range(self.n_actions) if i != action]
            action = np.random.choice(remain_actions)
        new_state, reward, done, info = self.env.step(action)
        new_state = self.map_obs(new_state)
        return new_state, reward, done, info
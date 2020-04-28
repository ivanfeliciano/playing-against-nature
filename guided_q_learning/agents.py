# -*- coding: utf-8 -*-
import random
import os
import time

import numpy as np
import matplotlib.pyplot as plt


class QLearningAgent(object):
    """
    Clase base para un agente de Q learning.
    Aquí se configura el ambiente, los parámetros y
    el flujo del aprendizaje.
    """
    def __init__(self, environment, policy, causal=False, episodes=100, alpha=0.8, gamma=0.95, mod_episode=1):
        self.env = environment
        self.policy = policy
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.Q = self.env.init_q_table()
        self.mod_episode = mod_episode
        self.training = True
        self.causal = causal
    def select_action(self, state):
        if self.training:
            return self.policy.select_action(state, self.Q)
        return self.policy.select_action(state, self.Q, False)
    def train(self):
        self.training = True
        self.avg_reward = []
        rewards_per_episode = []
        for episode in range(self.episodes):
            total_episode_reward = 0
            state = self.env.reset()
            done = False
            step = 0
            while not done:
                action = self.select_action(state)
                new_state, reward, done, info = self.env.step(action)
                self.Q[state][action] = self.Q[state][action] + self.alpha * \
                                        (reward + self.gamma * np.max(self.Q[new_state]) -\
                                        self.Q[state][action])
                state = new_state
                total_episode_reward += reward
                step += 1
            rewards_per_episode.append(total_episode_reward)
            if episode == 0 or (episode + 1) % self.mod_episode == 0:
                rewards_per_episode = self.update_avg_reward(rewards_per_episode)
        return self.avg_reward
    def update_avg_reward(self, rewards_per_episode):
        episodes_block_avg_rwd = np.mean(rewards_per_episode)
        self.avg_reward.append(episodes_block_avg_rwd)
        return []
    def test(self):
        self.training = False
        rewards_per_episode = []
        self.avg_reward = []
        for episode in range(self.episodes):
            total_episode_reward = 0
            state = self.env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                new_state, reward, done, info = self.env.step(action)
                state = new_state
                total_episode_reward += reward
            rewards_per_episode.append(total_episode_reward)
            if episode == 0 or  (episode + 1) % self.mod_episode == 0:
                rewards_per_episode = self.update_avg_reward(rewards_per_episode)
        return self.avg_reward
    def plot_avg_reward(self, filename="average_reward"):
        x_axis = self.mod_episode * (np.arange(len(self.avg_reward)))
        plt.plot(x_axis, self.avg_reward)
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.savefig("plots/{}.png".format(filename))     
        plt.close()
    def get_training_avg_reward(self):
        return self.avg_reward


def main():
    for structure in ["one_to_one", "one_to_many", "many_to_one"]:
        for size in [5, 7, 9]:
            q = QLearning(episodes=1000, num=size, structure=structure)
            average_reward = q.train()
            q.test(episodes=100)

if __name__ == '__main__':
    main()


# -*- coding: utf-8 -*-
import random
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from light_env import LightEnv

MOD_EPISODE = 20

def powerset(n):
	powerset = []
	for i in range(1 << n):
		bitset = [0 for i in range(n)]
		for j in range(n):
			bitset[j] = 1 if (i & (1 << j)) else 0
		powerset.append(tuple(bitset))
	return powerset

class QLearning(object):
	"""docstring for QLearning"""
	def __init__(self, episodes=100, alpha=0.8, gamma=0.95, num=5,\
		structure="one_to_one", eps_test=0.1, filename="QLearning"):
		self.episodes = episodes
		self.alpha = alpha
		self.gamma = gamma
		self.env = LightEnv(horizon=num, num=num, structure=structure)
		self.env.keep_struct = False
		self.env.reset()
		self.env.keep_struct = True
		self.init_causal_struct = False
		self.number_of_actions = self.env.action_space.n
		self.number_of_states = self.env.action_space.n - 1
		self.Q = dict()
		self.init_q_matrix()
		self.eps_min = 0.1
		self.eps_max = 1.0
		self.training = True
		self.nb_steps = 0
		self.eps_test= eps_test
		self.true_action_prob = 0.9
		self.horizon = num
		self.structure = structure
		self.plot_name = filename
	def init_q_matrix(self):
		all_states = powerset(self.number_of_states)
		for state in all_states:
			self.Q[state] = np.zeros(self.number_of_actions)
	def get_current_value(self):
		if self.training:
			# Linear annealed: f(x) = ax + b.
			a = -float(self.eps_max - self.eps_min) / float(self.horizon)
			b = float(self.eps_max)
			value = max(self.eps_min, a * float(self.step) + b)
		else:
			value = self.eps_test
		return value
	def action_selection(self, state):
		r = np.random.uniform()
		eps = self.get_current_value()
		if r > eps:
			return np.argmax(self.Q[state])
		return self.env.action_space.sample()
	def train(self, use_reward_feedback=False, stochastic=False):
		# print("TRAINING")
		state = self.env.reset()
		optimal_reward_episodes = []
		list_of_individual_episode_reward = []
		avg_reward_all_training = []
		for episode in range(self.episodes):
			reward_episode = 0
			state = tuple(self.env.reset()[:self.number_of_states].astype(int))
			done = False
			self.step = 1
			while not done:
				action = self.action_selection(state)
				if stochastic and np.random.uniform() > self.true_action_prob:
						remain_actions = [i for i in range(self.number_of_actions)]
						remain_actions.remove(action)
						action = np.random.choice(remain_actions)
				new_state, reward, done, info = self.env.step(action)
				new_state = tuple(new_state[:self.number_of_states].astype(int))
				self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state]) - self.Q[state][action])
				state = new_state
				reward_episode += reward
				self.step += 1
			list_of_individual_episode_reward.append(reward_episode)
			if episode == 0 or (episode + 1) % MOD_EPISODE == 0:
				ave_reward = np.mean(list_of_individual_episode_reward)
				# print("Episode : {}, Avg reward : {}".format(episode, ave_reward))
				avg_reward_all_training.append(ave_reward)
				list_of_individual_episode_reward = []
		# plot_x_axis = MOD_EPISODE * (np.arange(len(avg_reward_all_training)) + 1)
		# plt.plot(plot_x_axis, avg_reward_all_training, label=self.plot_name)
		# plt.xlabel('Episodes')
		# plt.ylabel('Average Reward')
		# plt.legend()
		# plt.title('Average Reward Comparison')
		# plt.savefig(self.plot_name + "_training_{}_{}.jpg".format(self.structure, self.number_of_states))     
		# plt.close()
		return avg_reward_all_training, optimal_reward_episodes
	def test(self, episodes=20, stochastic=False):
		print("TEST")
		self.training = False
		optimal_reward_episodes = []
		list_of_individual_episode_reward = []
		avg_reward_all_training = []
		for episode in range(episodes):
			reward_episode = 0
			state = tuple(self.env.reset()[:self.number_of_states].astype(int))
			done = False
			self.step = 1
			while not done:
				action = self.action_selection(state)
				if stochastic and np.random.uniform() > self.true_action_prob:
						remain_actions = [i for i in range(self.number_of_actions)]
						remain_actions.remove(action)
						action = np.random.choice(remain_actions)
				new_state, reward, done, info = self.env.step(action)
				state = tuple(new_state[:self.number_of_states].astype(int))
				reward_episode += reward
			list_of_individual_episode_reward.append(reward_episode)
			if episode == 0 or (episode + 1) % MOD_EPISODE == 0:
				ave_reward = np.mean(list_of_individual_episode_reward)
				# print("Episode : {}, Avg reward : {}".format(episode, ave_reward))
				avg_reward_all_training.append(ave_reward)
				list_of_individual_episode_reward = []
		# plot_x_axis = MOD_EPISODE * (np.arange(len(avg_reward_all_training)) + 1)
		# plt.plot(plot_x_axis, avg_reward_all_training, label=self.plot_name)
		# plt.xlabel('Episodes')
		# plt.ylabel('Average Reward')
		# plt.legend()
		# plt.title('Average Reward Comparison')
		# plt.savefig(self.plot_name + "_test_{}_{}.jpg".format(self.structure, self.number_of_states))     
		# plt.close()
		return avg_reward_all_training, optimal_reward_episodes

def main():
	for structure in ["one_to_one", "one_to_many", "many_to_one"]:
		for size in [5, 7, 9]:
			q = QLearning(episodes=1000, num=size, structure=structure)
			average_reward = q.train()
			q.test(episodes=100)

if __name__ == '__main__':
	main()


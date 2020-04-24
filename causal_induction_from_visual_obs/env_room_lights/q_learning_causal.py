# -*- coding: utf-8 -*-
import random
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from light_env import LightEnv
from q_learning import QLearning

class QlearningAssisted(QLearning):
	"""docstring for QlearningAssisted"""
	def action_selection(self, state):
		r = np.random.uniform()
		eps = self.get_current_value()
		if r > eps:
			return np.argmax(self.Q[state])
		r = np.random.uniform()
		if r > 0.8:
			return self.env.action_space.sample()
		causal_structure = self.get_causal_structure()
		goal = self.env.goal
		macro_state = self.env.state
		targets = []
		for i in range(len(goal)):
			if goal[i] == 1 and macro_state[i] == 0:
				targets.append(i)
		random.shuffle(targets)
		for target in targets:
			nonzero = [i for i in range(len(causal_structure[:, target])) if causal_structure[i][target] == 1]
			random.shuffle(nonzero)
			if len(nonzero) > 0:
				return nonzero[0]
		return self.env.action_space.sample()
	def get_causal_structure(self):
		if not self.init_causal_struct:
			self.init_causal_struct = True
			self.causal_structure = self.env.aj
		return self.causal_structure


class QlearningWrongAssited(QlearningAssisted):
	"""docstring for QlearningAssisted"""
	def get_causal_structure(self):
		if not self.init_causal_struct:
			self.init_causal_struct = True
			self.causal_structure = self.env.aj
			for i in range(len(self.causal_structure)):
				for j in range(len(self.causal_structure)):
						if np.random.uniform() > 0.75:
							self.causal_structure[i][j] = 0 if self.causal_structure[i][j] == 1 else 1 
		return self.causal_structure


class QlearningPartiallyAssited(QlearningAssisted):
	"""docstring for QlearningAssisted"""
	def get_causal_structure(self):
		if not self.init_causal_struct:
			self.init_causal_struct = True
			self.causal_structure = self.env.aj
			for i in range(len(self.causal_structure)):
				for j in range(len(self.causal_structure)):
					if self.causal_structure[i][j] and np.random.uniform() > 0.75:
						self.causal_structure[i][j] = 0
		return self.causal_structure


def main():
	for structure in ["one_to_one", "one_to_many", "many_to_one"]:
		for size in [5, 7, 9]:
			q = QlearningAssisted(episodes=1000, num=size, structure=structure, filename="QlearningAssisted")
			average_reward = q.train()
			q.test(episodes=100)
if __name__ == '__main__':
	main()
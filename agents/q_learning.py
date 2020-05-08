import logging
import numpy as np

from agents.core import Agent

class QLearning(Agent):
	"""
	Q learning para un bandido con una sola acción.
	"""
	def __init__(self, nature, epsilon):
		super().__init__(nature)
		self.epsilon = epsilon
		self.action_variable = self.nature.model.get_intervention_variables()
		self.action_values = self.nature.model.get_variable_values(self.action_variable[0])
		self.target = self.nature.model.get_target_variable()
		self.k = np.zeros(len(self.action_values), dtype=np.int)
		self.Q = np.zeros(len(self.action_values), dtype=np.float)
	
	def update_q(self, action, reward):
		self.k[action] += 1
		self.Q[action] += (1. / self.k[action]) * (reward - self.Q[action])
	
	def make_decision(self):
		r = np.random.random()
		if r < self.epsilon:
			return np.random.randint(len(self.action_values))
		action = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
		return action

	def training(self, rounds):
		for i in range(rounds):
			action = self.make_decision()
			state = self.nature.action_simulator(self.action_variable, [action])
			reward = state[self.target]
			self.update_q(action, reward)
			self.rewards_per_round.append(reward)
		logging.info(self.rewards_per_round)
		return self.rewards_per_round


if __name__ == "__main__":
	from true_causal_model import TrueCausalModel
	from model import BaseModel
	logging.basicConfig(filename='logs/qlearningAgent.log',
						filemode='w', level=logging.INFO)
	model = BaseModel('configs/model_parameters.json')
	nature = TrueCausalModel(model)
	rounds = 100
	qlearning_agent = QLearning(nature, 0.3)
	qlearning_agent.training(rounds)

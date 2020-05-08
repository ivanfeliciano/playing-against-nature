import logging
import numpy as np

from agents.core import Agent


class RandomAgent(Agent):
	"""
	Q learning para un bandido con una sola acción.
	"""
	def make_decision(self):
		actions = tuple()
		for action_var in self.nature.model.get_intervention_variables():
			values = self.nature.model.get_variable_values(action_var)
			a = np.random.choice(values)
			actions += (a,)
		return actions

	def training(self, rounds):
		target = self.nature.model.get_target_variable()
		for i in range(rounds):
			actions = self.make_decision()
			state = self.nature.action_simulator(
				self.nature.model.get_intervention_variables(), actions)
			reward = state[target]
			self.rewards_per_round.append(reward)
		logging.info(self.rewards_per_round)
		return self.rewards_per_round


if __name__ == "__main__":
	from true_causal_model import TrueCausalModel
	from model import BaseModel
	logging.basicConfig(filename='logs/randomAgent.log',
					 filemode='w', level=logging.INFO)
	model = BaseModel('configs/model_parameters.json')
	nature = TrueCausalModel(model)
	rounds = 10
	random_agent = RandomAgent(nature)
	random_agent.training(rounds)

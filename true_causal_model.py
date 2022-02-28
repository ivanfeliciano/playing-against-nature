import logging

import numpy as np

from model import BaseModel


class TrueCausalModel:
	"""
	El objeto de TrueCausalModel contiene el modelo causal de la naturaleza.

	Args:
		model (BaseModel) : un objeto de la clase BaseModel que contiene al modelo
		gráfico probabilista.

	"""
	def __init__(self, model):
		self.model = model

	def action_simulator(self, chosen_actions, values_chosen_actions):
		"""
		Ejecuta las acciones en el modelo causal verdadero (Nature).

		Args:
			chosen_actions (list): lista de las variables de accción a las que se les asignan
			valores.
			values_chosen_actions (list): lista de los valores que se asignan a las variables de acción.
		
		to-do tal vez sería mejor recibir como parámetro un diccionario
		"""
		response = dict()
		elements = None
		probabilities = None
		for nat_var in self.model.get_nature_variables():
			probabilities = self.model.get_nature_var_prob(nat_var)
			elements = [i for i in range(len(probabilities))]
			res = np.random.choice(elements, p=probabilities)
			response[nat_var] = res
		for idx, variable in enumerate(chosen_actions):
			response[variable] = values_chosen_actions[idx]
		ordered_variables = self.model.get_graph_toposort()
		ordered_variables = [i for i in ordered_variables\
							if i not in response]
		for unknown_variable in ordered_variables:
			if not response.get(unknown_variable):
				response[unknown_variable] = self.model.make_inference(unknown_variable,\
					response)
		return response

class TrueCausalModelEnv(TrueCausalModel):
	def __init__(self, model):
		self.model = model
		self.reward = None
		self.num = len(self.model.get_intervention_variables())
		self.cause_val = {"cause_{}".format(i) : 
										0 for i in range(self.num)}
		self.state = None
		self.reward = 0
	def reset(self):
		self.cause_val = {"cause_{}".format(i):
					0 for i in range(self.num)}
		self.state = None
		self.reward = 0
	def action_simulator(self, env, chosen_action):
		"""
		Ejecuta un paso en el ambiente y genera respuesta.
		"""
		action = int(chosen_action.strip().split("_")[-1])
		state, reward, done, info = env.step(action)
		self.state = state
		self.reward += reward
		response = dict()
		response["reward"] = reward
		response["done"] = done
		if action < env.num:
			# self.cause_val["cause_{}".format(action)] += 1 
			# self.cause_val["cause_{}".format(action)] %= 2 
			response["cause_{}".format(action)] = 1
		for i in range(env.num):
			response["effect_{}".format(i)] = int(state[i])
			if i != action:
				response["cause_{}".format(i)] = 0
		# for cause in self.cause_val:
		# 	response[cause] = self.cause_val[cause]
		return response
def main():
	logging.basicConfig(filename="logs/test_nature_light.log", filemode='w', level=logging.INFO)
	model = BaseModel('configs/model_parameters.json')
	tcm = TrueCausalModel(model)
	tcm.model.save_pgm_as_img("gt-graph.png")
	for cpd in tcm.model.pgmodel.get_cpds():
		print(cpd)
	model = BaseModel('configs/model_parameters_reaction.json')
	tcm = TrueCausalModel(model)
	tcm.model.save_pgm_as_img("gt-graph-reaction.png")
	for cpd in tcm.model.pgmodel.get_cpds():
		print(cpd)
	model = BaseModel('configs/model_parameters_lives.json')
	tcm = TrueCausalModel(model)
	tcm.model.save_pgm_as_img("gt-graph-lives.png")
	for cpd in tcm.model.pgmodel.get_cpds():
		print(cpd)
	# r = tcm.action_simulator(['Tratamiento'], [1])
	# print(r)
	# r = tcm.action_simulator(['Tratamiento'], [0])
	# print(r)
	# from utils.light_env_utils import generate_model_from_env
	# from env.light_env import LightEnv
	# env = LightEnv(structure="one_to_one")
	# env.keep_struct = False
	# env.reset()
	# env.keep_struct = True
	# model = generate_model_from_env(env)
	# nature_light_switch = TrueCausalModelEnv(model)
	# variable = "cause_1"
	# value = 1
	# r = nature_light_switch.action_simulator(env, variable)
	# print(r)


if __name__ == '__main__':
	main()

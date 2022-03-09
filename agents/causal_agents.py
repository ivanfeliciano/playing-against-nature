import logging
import json	
import itertools
from copy import deepcopy

import numpy as np
import networkx as nx
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from scipy.stats import beta, dirichlet
import matplotlib.pyplot as plt

from agents.core import Agent

class CausalAgent(Agent):
	def __init__(self, nature, pgmodel):
		super().__init__(nature)
		self.beliefs = dict()
		self.model = deepcopy(pgmodel)
	def do_calculus(self, target, intervened_variables):
		"""
		Calcula de probabilidad condicional de target dados
		los valores en las variables intervenidas.
		"""
		return self.model.conditional_probability(target, \
					intervened_variables).values
	def make_decision(self, target, intervened_variables):
		"""
		Elige la mejor combinación acciones que puede tomar
		de acuerdo con la mayor probilidad de obtener cierto
		valor en el target. Por ahora sólo funciona para una variable
		target y múltiples variables intervenidas.
		"""
		target_name = target["variable"]
		target_value = int(target["value"])
		val_inter_vars = [self.model.get_variable_values(i)\
							for i in intervened_variables]
		cartesian_prod = list(itertools.product(*val_inter_vars))
		idx = np.random.choice(len(cartesian_prod))
		best_actions = cartesian_prod[idx]
		max_prob = -float("inf")

		for vars_tuples in itertools.product(*val_inter_vars):
			query_dict = dict()
			for i in range(len(intervened_variables)):
				query_dict[intervened_variables[i]] = vars_tuples[i]
			prob_table = self.do_calculus(target_name, query_dict)
			logging.info(prob_table)
			prob = prob_table[target_value]
			if prob > max_prob: max_prob = prob; best_actions = vars_tuples
		return best_actions
	def training(self, rounds, target_value):
		raise NotImplementedError

	def make_decision_advanced(self, target, intervened_variables, threshold=-float("inf")):
		"""
		Elige la mejor combinación acciones que puede tomar
		de acuerdo con la mayor probilidad de obtener cierto
		valor en el target. Por ahora sólo funciona para una variable
		targer y múltiples variables intervenidas.
		"""
		target_name = target["variable"]
		target_value = int(target["value"])
		val_inter_vars = [self.model.get_variable_values(i)\
							for i in intervened_variables]
		cartesian_prod = list(itertools.product(*val_inter_vars))
		best_actions = None
		max_prob = threshold
		print(val_inter_vars)
		for vars_tuples in itertools.product(*val_inter_vars):
			query_dict = dict()
			for i in range(len(intervened_variables)):
				query_dict[intervened_variables[i]] = vars_tuples[i]
			prob_table = self.do_calculus(target_name, query_dict)
			logging.info(prob_table)
			prob = prob_table[target_value]
			if prob >= max_prob: max_prob = prob; best_actions = vars_tuples
		return best_actions
class FullyInformedAgent(CausalAgent):
	def training(self, rounds, target_value):
		intervention_vars = self.model.get_intervention_variables()
		target = {
			"variable": self.model.get_target_variable(),
			"value" : target_value
			}
		for i in range(rounds):
			best_actions = self.make_decision(target, intervention_vars)
			logging.info("Best actions {} {}".format(intervention_vars, best_actions))
			nature_response = self.nature.action_simulator(intervention_vars, \
								best_actions)
			logging.info(nature_response)
			self.rewards_per_round.append(nature_response[target["variable"]])
		return self.rewards_per_round	

class HalfBlindAgent(CausalAgent):
	"""
	Un objeto de esta clase simula a un agente que tiene información parcial del 
	"""
	def __init__(self, nature, pgmodel):
		super().__init__(nature, pgmodel)
		self.alpha_params = dict()
		self.init_alpha_and_beliefs()

	def init_alpha_and_beliefs(self):
		"""
		Inicializa los parámetros alpha y los de de conteo que se usan en la distribución Dirichlet.
		Los parámetros se guardan en su diccionario correspondiente. Por cada variable se usa como llave una cadena en orden lexicográfico  de sus padres (si tiene) y valores.
		
		Para alpha la estructura es como la siguiente:
		
		{
			"variable" : {
				"has_parents" : boolean,
				"parent1_0, ... , parentn_1" : float
			}
		}
		"""
		logging.info("Initializing alpha parameters")
		adj_list = self.model.get_nodes_and_predecessors()
		logging.info(json.dumps(adj_list, indent=2))
		vars_possible_values = {n : \
			self.model.get_variable_values(n) for n in adj_list}
		
		for node in adj_list:
			node_object = dict()
			parents_to_string = ""
			combinations = itertools.product(*[
							vars_possible_values[p]
							for p in adj_list[node]]
							)
			node_object["has_parents"] = True if len(adj_list[node]) > 0 else False
			for combination in combinations:
				parents_to_string = ""
				for i in range(len(combination)):
					parents_to_string += "{}_{} ".format(adj_list[node][i], \
										combination[i])
				parents_to_string = parents_to_string.strip()
				k = len(vars_possible_values[node])
				alpha_value = np.random.rand(k)
				node_object[parents_to_string] = alpha_value.tolist()
			self.alpha_params[node] = deepcopy(node_object)
		logging.info(json.dumps(self.alpha_params, indent=2))
		self.update_beliefs()
		self.update_cpts_causal_model()
	def update_beliefs(self, observation_dict=None):
		"""
		Este método crea el esqueleto de las CPTs usando una distribución Dirichlet.
		Usa el diccionario alpha para crear las CPTs y guardarlas en el diccionario 
		de beliefs.
		"""
		# primero actualizo mis alphas
		if observation_dict:
			self.update_alpha_parameters(observation_dict)
		# recorrer el diccionario alpha
		for variable in self.alpha_params:
			table = []
			if not self.alpha_params[variable]["has_parents"]:
				alpha = self.alpha_params[variable][""]
				table = dirichlet.rvs(alpha, size=1).tolist()
			else:
				for parents_instance in self.alpha_params[variable]:
					if parents_instance == "has_parents":
						continue
					alpha = self.alpha_params[variable][parents_instance]
					probabilities = np.squeeze(dirichlet.rvs(alpha, size=1))
					table.append(probabilities)
				table = np.array(table).transpose().tolist()
			self.beliefs[variable] = table
		logging.info("Beliefs after update")
		logging.info(json.dumps(self.beliefs, indent=2))
	def update_cpts_causal_model(self):
		"""
		Con este método se crean las CPTs usando la biblioteca pgmpy a partir
		del diccionario de beliefs que tiene el agente. 
		"""
		adj_list = self.model.get_nodes_and_predecessors()
		logging.info("Updating cpts from beliefs")
		var_values = {n : \
			self.model.get_variable_values(n) for n in adj_list}
		backup_model = self.model.pgmodel.copy()
		for variable in self.beliefs:
			evidence = adj_list[variable]
			evidence_card = [len(var_values[parent]) for parent in evidence]
			cpd_table = TabularCPD(variable=variable, variable_card=\
						len(var_values[variable]), values=self.beliefs[variable],\
						evidence=evidence, evidence_card=evidence_card)
			self.model.pgmodel.add_cpds(cpd_table)
		if self.model.pgmodel.check_model():
			self.model.infer_system = VariableElimination(self.model.pgmodel)
			# logging.info("PGMPY CPTs")
			# for cpd in self.model.pgmodel.get_cpds():
			# 	logging.info(cpd)
		else:
			for cpd in backup_model.get_cpds():
				logging.info(cpd)
			raise ValueError("Error with CPTs")
	def update_alpha_parameters(self, observation_dict):
		"""
		En este método actualizo mi diccionario de creencias (beliefs) 
		de acuerdo con lo que observó el agente.
		"""
		adj_list = self.model.get_nodes_and_predecessors()
		
		for node in observation_dict:
			node_object = dict()
			parents_to_string = ""
			node_value = observation_dict[node]
			for parent in adj_list[node]:
				parents_to_string += "{}_{} ".format(parent, observation_dict[parent])
			parents_to_string = parents_to_string.strip()
			self.alpha_params[node][parents_to_string][node_value] += 1

		# logging.info("ALPHAS after counting update")
		# logging.info(json.dumps(self.alpha_params, indent=2))
	def make_decision_using_obs(self, target, intervened_variables, obs):
		target_name = target["variable"]
		target_value = int(target["value"])
		val_inter_vars = [self.model.get_variable_values(i)\
							for i in intervened_variables]
		cartesian_prod = list(itertools.product(*val_inter_vars))
		idx = np.random.choice(len(cartesian_prod))
		best_actions = cartesian_prod[idx]
		max_prob = -float("inf")

		for vars_tuples in itertools.product(*val_inter_vars):
			query_dict = dict()
			for i in range(len(intervened_variables)):
				query_dict["X"] = obs["X"]
				query_dict["Y"] = obs["Y"]
				query_dict[intervened_variables[i]] = vars_tuples[i]
			prob_table = self.do_calculus(target_name, query_dict)
			logging.info(prob_table)
			prob = prob_table[target_value]
			if prob > max_prob: max_prob = prob; best_actions = vars_tuples
		return best_actions
	def training(self, rounds, target_value):
		intervention_vars = self.model.get_intervention_variables()
		target = {
			"variable": self.model.get_target_variable(),
			"value" : target_value
			}
		for i in range(rounds):
			self.update_beliefs({"X" : 0, "Y" : 1})
			self.update_cpts_causal_model()
			best_actions = self.make_decision(
				target, intervention_vars)
			# logging.info("Best actions {} {}".format(intervention_vars, best_actions))
			nature_response = self.nature.action_simulator(intervention_vars, \
								best_actions)
			# logging.info(nature_response)
			self.rewards_per_round.append(nature_response[target["variable"]])
			self.update_beliefs(nature_response)
			self.update_cpts_causal_model()
		for table in self.model.pgmodel.get_cpds():
			logging.info(table.values)
			logging.info(table.values.shape)
			logging.info(np.squeeze(table.values))
		return self.rewards_per_round
	def get_cpdts(self):
		return self.model.pgmodel.get_cpds()


def main():
	from true_causal_model import TrueCausalModel
	from model import BaseModel
	logging.basicConfig(filename='logs/causalAgent.log', filemode='w', level=logging.INFO)
	model = BaseModel('configs/model_parameters.json')
	nature = TrueCausalModel(model)
	rounds = 100
	target_value = 1
	half_blind_agent = HalfBlindAgent(nature, model)
	half_blind_agent.training(rounds, target_value)
	# logging.info("FULL AGENT")
	# full_agent = FullyInformedAgent(nature, model)
	# full_agent.training(rounds, target_value)


	
if __name__ == '__main__':
	main()

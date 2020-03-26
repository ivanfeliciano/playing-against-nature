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

from true_causal_model import TrueCausalModel
from model import BaseModel

class HalfBlindAgent(object):
	"""
	Un objeto de esta clase simula a un agente que tiene información parcial del 
	"""
	def __init__(self, pgmodel):
		self.model = None
		self.partial_model = pgmodel
		self.alpha_params = dict()
		self.counting_params = dict()
		self.beliefs = dict()
		self.reward_per_round = []
		self.n_rounds = 0
		self.nature = TrueCausalModel(pgmodel)
		self.init_alpha_and_counting_params()

	def init_alpha_and_counting_params(self):
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

		Para los contadores tenemos lo mismo pero con los respectivos valores de cada variable:

		{
			"variable_vi" : {
				"has_parents" : boolean,
				"parent1_0, ... , parentn_1" : int
			},
			"variable_vj" : {
				"has_parents" : boolean,
				"parent1_0, ... , parentn_1" : int
			}
		}

		"""
		adj_list = self.partial_model.get_nodes_and_predecessors()
		print(json.dumps(adj_list, indent=2))
		vars_possible_values = {n : [0, 1] for n in adj_list}
		# to do integrar esta función
		# vars_possible_values = self.partial_model.get_number_of_values()
		
		for node in adj_list:
			object = dict()
			parents_to_string = ""
			combinations = itertools.product(*[
							vars_possible_values[p]
							for p in adj_list[node]]
							)
			object["has_parents"] = True if len(adj_list[node]) > 0 else False
			for combination in combinations:
				parents_to_string = ""
				for i in range(len(combination)):
					parents_to_string += adj_list[node][i] + "_" +\
											str(combination[i]) + " "
				parents_to_string = parents_to_string.strip()
				k = len(vars_possible_values[node])
				alpha_value = np.random.rand(k)
				object[parents_to_string] = alpha_value.tolist()
			self.alpha_params[node] = deepcopy(object)
			for value in vars_possible_values[node]:
				for key in object:
					if key != "has_parents": object[key] = 0
				self.counting_params[node + "_" + str(value)] = object
		print("ALPHAS")
		print(json.dumps(self.alpha_params, indent=2))
		print("CONTADORES")
		print(json.dumps(self.counting_params, indent=2))
	def generate_dirichlet_cpts(self):
		"""
		Este método crea el esqueleto de las CPTs usando una distribución Dirichlet.
		Usa el diccionario alpha para crear las CPTs.
		"""

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
			print("Rough CPTs")
			print(json.dumps(self.beliefs, indent=2))
	def update_cpts_causal_model(self):
		"""
		Con este método se crean las CPTs usando la biblioteca pgmpy a partir
		del diccionario de beliefs que tiene el agente. 
		"""
		adj_list = self.partial_model.get_nodes_and_predecessors()
		var_values = {n : [0, 1] for n in adj_list}
		for variable in self.beliefs:
			evidence = adj_list[variable]
			evidence_card = [len(var_values[parent]) for parent in evidence]
			cpd_table = TabularCPD(variable=variable, variable_card=\
						len(var_values[variable]), values=self.beliefs[variable],\
						evidence=evidence, evidence_card=evidence_card)
			print(cpd_table)
def main():
	model = BaseModel('model_parameters.json')
	test_half_blind_agent = HalfBlindAgent(model)
	test_half_blind_agent.generate_dirichlet_cpts()
	test_half_blind_agent.update_cpts_causal_model()

if __name__ == '__main__':
	main()
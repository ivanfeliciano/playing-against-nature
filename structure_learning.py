#!/usr/bin/env python
# coding: utf-8

import itertools
from copy import deepcopy, copy
import logging
from multiprocessing import Pool

import numpy as np
import networkx as nx
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from scipy.stats import beta, dirichlet
import matplotlib.pyplot as plt
import pandas as pd


from model import BaseModel
from true_causal_model import TrueCausalModel, TrueCausalModelEnv
from agents.causal_agents import HalfBlindAgent
from utils.vis_utils import plot_measures, plot_probabilities
from env.light_env import LightEnv
from utils.light_env_utils import *
from utils.helpers import *
from utils.modified_estimator import SmoothedMaximumLikelihoodEstimator
np.random.seed(0)

def create_pij(variables, causal_order, invalid_edges):
	"""
	Inicializa un diccionario que contiene las creencias
	de conexión.
	{
		(var_i, var_j) : float,
	}
	"""
	connection_tables = dict()
	for pair in itertools.combinations((variables), 2):
		# proba = np.random.rand()
		proba = 0.5
		if is_a_valid_edge(pair[0], pair[1], causal_order, invalid_edges):
			connection_tables[(pair[0], pair[1])] = proba
		elif is_a_valid_edge(pair[1], pair[0], causal_order, invalid_edges):
			connection_tables[(pair[1], pair[0])] = proba
	return connection_tables

def create_graph_from_beliefs(variables, connection_tables):
	"""
	Retorna una lista de adyacencia a partir de la creencias
	de conexión.
	"""
	adj_list = dict()
	for variable in variables:
		adj_list[variable] = []
	for edge in connection_tables:
		r = np.random.rand()
		if r <= connection_tables[edge]:
			adj_list[edge[0]].append(edge[1])
	return adj_list

def adj_list_to_ebunch_and_nodes(adj_list):
	"""
	Convierte la lista de adyacencia a una lista de aristas y de nodos
	que sirven para crear un modelo de pgmpy.
	"""
	nodes = []
	ebunch = []
	for node in adj_list:
		nodes.append(node)
		for child in adj_list[node]:
			ebunch.append((node, child))
	return ebunch, nodes

def explore_and_generate_data(nature, intervention_vars, n_steps=100):
	"""
	Crea un diccionario con las observaciones a través de la interacción
	con la naturaleza. La naturaleza es un objeto de la clase TrueCausalModel.
	"""
	data_dict = dict()
	for i in range(n_steps):
		best_actions = np.random.randint(2, size=1)
		idx_action = np.random.randint(len(intervention_vars))
		action = intervention_vars[idx_action]
		obs = nature.action_simulator([action], best_actions)
		for k in obs:
			if k not in data_dict:
				data_dict[k] = []
			data_dict[k].append(obs[k])
	return data_dict

def update_prob_measures(connection_probas, connection_tables):
	"""
	Va actualizando un diccionario con las probabilidades de conexión entre las 
	variables. Las llaves son las aristas y los valores son listas con las probabilidades
	de conexión según las creencias en cada paso de aprendizaje.
	"""
	for k in connection_tables:
		if k not in connection_probas:
			connection_probas[k] = []
		connection_probas[k] += [connection_tables[k]]

def generate_approx_model_from_graph(ebunch, nodes, df):
	"""
	Aprende un modelo Bayesiano de pgmpy usando un datos de un
	dataframe de pandas. Primero se hace un barajado de los datos.
	"""
	df = df.sample(frac=1)
	approx_model = BayesianModel(ebunch)
	approx_model.add_nodes_from(nodes)
	state_names = dict()
	for pair in ebunch:
		state_names[pair[0]] = [0, 1]
		state_names[pair[1]] = [0, 1]
	for node in nodes:
		state_names[node] = [0, 1]
	approx_model.fit(df, state_names=state_names, estimator=SmoothedMaximumLikelihoodEstimator)
	return approx_model


def pool_handler(pair, ebunch, nodes, df, nature_response, modified_model):
	"""
	Función que genera un modelo a partir de las observaciones y calcula
	la función de probabilidad conjunta para una obseravación en 
	específico.
	
	"""
	pgmodel = generate_approx_model_from_graph(ebunch, nodes, df)
	modified_model.reset(pgmodel, ebunch, nodes)
	proba = modified_model.get_joint_prob_observation(nature_response)
	return (pair, proba)

def update_connection_beliefs(model, connection_tables, df, nature_response):
	"""
	Función para actualizar las creencias usando 
	un multiprocessing. Utiliza un modelo del agente acerca del mundo,
	las probabilidades de conexión, datos de interacciones en un dataframe y la observación
	actual enviada por la naturaleza.

	Para actualizar cada conexión se utiliza la regla:

	p'ij ← P(o| Mij) Probabilidad de la observación dado un modelo con la conexión entre i->j
	p'~ij <- P(o| M~ij) Probabilidad de la observación dado un modelo sin la conexión entre i->j
	pij ← (pij * p'ij) / (pij * p'ij + (1 - pij) * p'~ij)
	"""
	base_model = deepcopy(model)
	modified_model = deepcopy(model)
	ebunch = model.get_ebunch()
	nodes = model.get_nodes()
	base_model_from_data = generate_approx_model_from_graph(ebunch, nodes, df)
	base_model.reset(base_model_from_data, ebunch, nodes)
	p_obs_given_base_model = base_model.get_joint_prob_observation(
		nature_response)
	p_complements = []
	p_subs = []
	p_complements_pool = []
	p_subs_pool = []
	for pair in connection_tables:
		cause = pair[0]
		effect = pair[1]
		if pair in ebunch:
			p_sub = p_obs_given_base_model
			p_subs.append((pair, p_sub))
			ebunch_without_ij = copy(ebunch)
			ebunch_without_ij.remove((cause, effect))
			p_complements_pool.append(
				(pair, ebunch_without_ij, nodes, copy(df), copy(nature_response), deepcopy(model)))
		else:
			ebunch_with_ij = copy(ebunch)
			ebunch_with_ij.append((cause, effect))	
			key_ebunch_with_ij = tuple(sorted(ebunch_with_ij))
			p_complement = p_obs_given_base_model
			p_complements.append((pair, p_complement))
			p_subs_pool.append((pair, ebunch_with_ij, nodes, copy(
				df), copy(nature_response), deepcopy(model)))
	with Pool() as pool:
		p_complements += pool.starmap(pool_handler, p_complements_pool)
	with Pool() as pool:
		p_subs += pool.starmap(pool_handler, p_subs_pool)
	p_sub_dict = dict()
	p_complement_dict = dict()
	for pair in p_complements:
		p_complement_dict[pair[0]] = pair[1]
	for pair in p_subs:
		p_sub_dict[pair[0]] = pair[1]
	for pair in connection_tables:
		connection_tables[pair] = (connection_tables[pair] * p_sub_dict[pair]) / (p_sub_dict[pair] * connection_tables[pair] + p_complement_dict[pair] * (1 - connection_tables[pair]))
	return connection_tables

def update_connection_beliefs_seq(model, connection_tables, df, nature_response):
	"""
	Función para actualizar las creencias usando 
	un proceso. Utiliza un modelo del agente acerca del mundo,
	las probabilidades de conexión, datos de interacciones en un dataframe y la observación
	actual enviada por la naturaleza.

	Para actualizar cada conexión se utiliza la regla:

	p'ij ← P(o| Mij) Probabilidad de la observación dado un modelo con la conexión entre i->j
	p'~ij <- P(o| M~ij) Probabilidad de la observación dado un modelo sin la conexión entre i->j
	pij ← (pij * p'ij) / (pij * p'ij + (1 - pij) * p'~ij)
	"""
	base_model = deepcopy(model)
	modified_model = deepcopy(model)
	ebunch = model.get_ebunch()
	nodes = model.get_nodes()
	base_model_from_data = generate_approx_model_from_graph(ebunch, nodes, df)
	base_model.reset(base_model_from_data, ebunch, nodes)
	p_obs_given_base_model = base_model.get_joint_prob_observation(
		nature_response)
	logging.info("Edges: {}".format(ebunch))
	for pair in connection_tables:
		cause = pair[0]
		effect = pair[1]
		logging.info("Updating {} -> {}".format(cause, effect))
		if pair in ebunch:
			logging.info("Edge in graph")
			logging.info("ebunch : {}".format(ebunch))
			p_sub = p_obs_given_base_model
			ebunch_without_ij = copy(ebunch)
			ebunch_without_ij.remove((cause, effect))
			pgmodel_notij = generate_approx_model_from_graph(ebunch_without_ij, nodes, df)
			modified_model.reset(pgmodel_notij, ebunch_without_ij, nodes)
			logging.info("Remove edge in graph")
			logging.info("ebunch withoutij: {}".format(ebunch_without_ij))
			p_complement = modified_model.get_joint_prob_observation(nature_response)
		else:
			logging.info("Edge not in graph")
			logging.info("ebunch withoutij : {}".format(ebunch))
			ebunch_with_ij = copy(ebunch)
			ebunch_with_ij.append((cause, effect))	
			key_ebunch_with_ij = tuple(sorted(ebunch_with_ij))
			p_complement = p_obs_given_base_model
			logging.info("Add edge in graph")
			logging.info("ebunch with : {}".format(ebunch_with_ij))
			pgmodel_ij = generate_approx_model_from_graph(ebunch_with_ij, nodes, df)
			modified_model.reset(pgmodel_ij, ebunch_with_ij, nodes)
			p_sub = modified_model.get_joint_prob_observation(nature_response)
		# print("Pij = {},  P~ij = {}".format(p_sub, p_complement))
		# print("{} * {} / ({} * {} + {} * (1 - {}))".format(connection_tables[pair], p_sub, p_sub, connection_tables[pair], p_complement, connection_tables[pair]))
		connection_tables[pair] = (connection_tables[pair] * p_sub) / (p_sub * connection_tables[pair] + p_complement * (1 - connection_tables[pair]))
	return connection_tables

def training(variables, rounds, connection_tables, data, unknown_model, nature):
	intervention_vars = nature.model.get_intervention_variables()
	connection_probas = dict()
	update_prob_measures(connection_probas, connection_tables)
	local_data = deepcopy(data)
	df = pd.DataFrame.from_dict(local_data)
	for rnd in range(rounds):
		idx_intervention_var = np.random.randint(len(intervention_vars))
		action = intervention_vars[idx_intervention_var]
		action_value = np.random.randint(2)
		nature_response = nature.action_simulator([action], [action_value])
		connection_tables = update_connection_beliefs(unknown_model, connection_tables, df, nature_response)
		update_prob_measures(connection_probas, connection_tables)
		for k in nature_response:
			local_data[k].append(nature_response[k])
		df = pd.DataFrame.from_dict(local_data)
		adj_list = create_graph_from_beliefs(variables, connection_tables)
		ebunch, nodes = adj_list_to_ebunch_and_nodes(adj_list)
		approx_model = generate_approx_model_from_graph(ebunch, nodes, df)
		unknown_model.reset(approx_model, ebunch, nodes)
	return connection_probas


def training_ligh_env_learning(variables, rounds, connection_tables, data_on, data_off, unknown_model_on, unknown_model_off, env):
	connection_probas = dict()
	local_data_on = deepcopy(data_on)
	local_data_off = deepcopy(data_off)
	df_on = pd.DataFrame.from_dict(local_data_on)
	df_off = pd.DataFrame.from_dict(local_data_off)
	update_prob_measures(connection_probas, connection_tables)
	for rnd in range(rounds):
		print("Round {} / {}".format(rnd + 1, rounds))
		action_idx = env.action_space.sample()
		action = "cause_{}".format(action_idx)
		nature_response = action_simulator(env, action)
		done = nature_response.pop("done", None)
		nature_response.pop("reward", None)
		change_to = nature_response.pop("change_to", None)
		if change_to == "on":
			connection_tables = update_connection_beliefs(
				unknown_model_on, connection_tables, df_on, nature_response)
		if change_to == "off":
			connection_tables = update_connection_beliefs(
				unknown_model_off, connection_tables, df_off, nature_response)
		else:
			if np.random.rand() < 0.5:
				connection_tables = update_connection_beliefs(
					unknown_model_off, connection_tables, df_on, nature_response)
			else:
				connection_tables = update_connection_beliefs(
					unknown_model_on, connection_tables, df_off, nature_response)
		update_prob_measures(connection_probas, connection_tables)
		adj_list = create_graph_from_beliefs(variables, connection_tables)
		ebunch, nodes = adj_list_to_ebunch_and_nodes(adj_list)
		if change_to == "on" or change_to == "nothing":
			for k in nature_response:
				local_data_on[k].append(nature_response[k])
			df_on = pd.DataFrame.from_dict(local_data_on)
			approx_model = generate_approx_model_from_graph(ebunch, nodes, df_on)
			unknown_model_on.reset(approx_model, ebunch, nodes)
		if change_to == "off" or change_to == "nothing":
			for k in nature_response:
				local_data_off[k].append(nature_response[k])
			df_off = pd.DataFrame.from_dict(local_data_off)
			approx_model = generate_approx_model_from_graph(ebunch, nodes, df_off)
			unknown_model_off.reset(approx_model, ebunch, nodes)
	return connection_probas

def light_env_learning(base_dir="results", structure="one_to_one", num=5, rounds=50, l=50, experiments=1, num_structures=10):
	from env.light_env import LightEnv
	rounds = rounds
	exploration_steps = l
	env = LightEnv(structure=structure, num=num)
	for s in range(num_structures):
		print("Structure {} / {}".format(s + 1, num_structures))
		env.keep_struct = False
		env.reset()
		env.keep_struct = True
		lights_on_model = generate_model_from_env(env)
		lights_off_model = generate_model_from_env(env, lights_off=True)
		variables = sorted(lights_on_model.get_graph_toposort())
		causal_order = variables
		invalid_edges = []
		causes = lights_on_model.get_intervention_variables()
		invalid_edges = generate_invalid_edges_light(variables, causes)
		experiments = 1
		global_results = dict()
		lights_on_model.save_digraph_as_img("{}/{}/light_env_{}.pdf".format(base_dir, structure, s))
		g_truth = {e : 1 for e in lights_on_model.digraph.edges}
		for i in range(experiments):
			connection_tables = create_pij(variables, causal_order, invalid_edges)
			adj_list = create_graph_from_beliefs(variables, connection_tables)
			ebunch, nodes = adj_list_to_ebunch_and_nodes(adj_list)
			data_on = dict()
			data_off = dict()
			data_on, data_off = explore_light_env(env, exploration_steps)
			df_on = pd.DataFrame.from_dict(data_on)
			df_off = pd.DataFrame.from_dict(data_off)
			approx_model_on = generate_approx_model_from_graph(ebunch, nodes, df_on)
			approx_model_off = generate_approx_model_from_graph(ebunch, nodes, df_off)
			unknown_model_on = deepcopy(lights_on_model)
			unknown_model_off = deepcopy(lights_off_model)
			unknown_model_on.reset(approx_model_on, ebunch, nodes)
			unknown_model_off.reset(approx_model_off, ebunch, nodes)
			connection_probs = training_ligh_env_learning(
				variables, rounds, connection_tables, data_on, data_off, unknown_model_on, unknown_model_off, env)
			for key in connection_probs:
				if key not in global_results:
					global_results[key] = []
				global_results[key].append(connection_probs[key])
		labels = []
		mean_vectors = []
		std_dev_vectors = []
		last_beliefs = dict()
		for key in global_results:
			mean_vec = np.mean(global_results[key], axis=0)
			labels += [key]
			mean_vectors.append(mean_vec)
			std_dev_vectors.append(np.std(global_results[key], axis=0))
			last_beliefs[key] = mean_vectors[-1][-1]
		x_axis = np.arange(len(mean_vectors[0]))
		# with open('mean.npy', 'wb') as f:
		# 	np.save(f, np.array(mean_vectors))
		# with open('std_dev.npy', 'wb') as f:
		# 	np.save(f, np.array(std_dev_vectors))
		# with open('labels.npy', 'wb') as f:
		# 	np.save(f, np.array(labels))
		plot_measures(x_axis, mean_vectors, std_dev_vectors, labels, "{}/{}/all_lights_struct_{}_exp_{}_rounds_{}".format(base_dir, structure, s, experiments, rounds), legend=False)
		for i in range(len(mean_vectors)):
			plot_measures(x_axis, [mean_vectors[i]], [std_dev_vectors[i]], [labels[i]], "{}/{}/lights_{}_struct_{}_exp_{}_rounds_{}".format(base_dir, structure, labels[i], s, experiments, rounds))
		for epsilon in [0, 0.25, 0.5, 0.75]:
			print("{} {}".format(compare_edges(g_truth, last_beliefs, epsilon), epsilon))
def basic_model_learning():
	DG = nx.DiGraph([("Reaction", "Lives"), ("Treatment", "Reaction"), ("Treatment", "Lives"), ("Disease", "Lives")])
	causal_order = list(nx.topological_sort(DG))
	# invalid_edges = [("Disease", "Treatment")]
	invalid_edges = []

	COMPLETE_MODEL = BaseModel('configs/model_parameters.json')
	nature = TrueCausalModel(COMPLETE_MODEL)
	variables = sorted(["Treatment", "Reaction", "Disease", "Lives"])
	intervention_vars = COMPLETE_MODEL.get_intervention_variables()
	target_value = 1
	target = {
		"variable": COMPLETE_MODEL.get_target_variable(),
		"value" : target_value
	}

	experiments = 10
	global_results = dict()
	rounds = 50
	n_exploration_steps = 10
	for i in range(experiments):
		data = explore_and_generate_data(nature, intervention_vars, n_steps=n_exploration_steps)
		df = pd.DataFrame.from_dict(data)
		connection_tables = create_pij(variables, causal_order, invalid_edges)
		adj_list = create_graph_from_beliefs(variables, connection_tables)
		ebunch, nodes = adj_list_to_ebunch_and_nodes(adj_list)
		approx_model = generate_approx_model_from_graph(ebunch, nodes, df)

		unknown_model = BaseModel('configs/incomplete_params.json')
		unknown_model.reset(approx_model, ebunch, nodes)
		unknown_model.show_graph()
		# agent = HalfBlindAgent(nature, unknown_model)

		connection_probas = training(
			variables, rounds, connection_tables, data, unknown_model, nature)
		print_dict(connection_tables)
		for key in connection_probas:
			if key not in global_results:
				global_results[key] = []
			global_results[key].append(connection_probas[key])
	labels = []
	mean_vectors = []
	std_dev_vectors = []
	for key in global_results:
		labels += [key]
		mean_vectors.append(np.mean(global_results[key], axis=0))
		std_dev_vectors.append(np.std(global_results[key], axis=0))
		print("{} {} {}".format(key, mean_vectors[-1][-1], std_dev_vectors[-1][-1]))
	x_axis = np.arange(len(mean_vectors[0]))
	plot_measures(x_axis, mean_vectors, std_dev_vectors, labels, "connection_beliefs_exp_{}_rounds_{}_{}".format(experiments, rounds, intervention_vars))
	for i in range(len(mean_vectors)):
		plot_measures(x_axis, [mean_vectors[i]], [std_dev_vectors[i]], [labels[i]], "connection_beliefs_{}_exp_{}_rounds_{}_{}".format(labels[i], experiments, rounds, intervention_vars))
	

if __name__ == '__main__':
	print("ONE TO ONE")
	light_env_learning(structure="one_to_one", l=20, num_structures=5, rounds=50)
	# print("ONE TO MANY")
	# light_env_learning(structure="one_to_many", l=20, num_structures=10, rounds=50)
	# print("MANY TO ONE")
	# light_env_learning(structure="many_to_one", l=20, num_structures=10, rounds=50)
	# print("ONE TO ONE")
	# light_env_learning(structure="one_to_one", l=200,
	#                    num_structures=10, rounds=20)
	# print("ONE TO MANY")
	# light_env_learning(structure="one_to_many", l=200,
	#                    num_structures=10, rounds=20)
	# print("MANY TO ONE")
	# light_env_learning(structure="many_to_one", l=200,
	#                    num_structures=10, rounds=20)
	# basic_model_learning()






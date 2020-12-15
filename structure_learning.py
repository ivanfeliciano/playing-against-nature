#!/usr/bin/env python
# coding: utf-8

import os
import time
import pickle
import itertools
import random
from copy import deepcopy, copy
import logging
from multiprocessing import Pool

import tqdm
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

from model import BaseModel
from true_causal_model import TrueCausalModel
from utils.vis_utils import plot_measures, plot_probabilities
from env.light_env import LightEnv
from utils.light_env_utils import *
from utils.helpers import *
from utils.modified_estimator import SmoothedMaximumLikelihoodEstimator
from policy import PolicyCM

np.random.seed(0)


def create_pij(variables, causal_order, invalid_edges, use_causal_order=True):
	"""
	Inicializa un diccionario que contiene las creencias
	de conexión.
	{
		(var_i, var_j) : float,
	}
	"""
	connection_tables = dict()
	for pair in itertools.combinations((variables), 2):
		proba = 0.5
		if use_causal_order:
			if is_a_valid_edge(pair[0], pair[1], causal_order, invalid_edges):
				connection_tables[(pair[0], pair[1])] = proba
			elif is_a_valid_edge(pair[1], pair[0], causal_order, invalid_edges):
				connection_tables[(pair[1], pair[0])] = proba
		else:
				connection_tables[(pair[0], pair[1])] = proba
				connection_tables[(pair[1], pair[0])] = proba
	return connection_tables

def create_graph_from_beliefs_unknown_order(variables, connection_tables):
	"""
	Retorna una lista de adyacencia a partir de la creencias
	de conexión.
	"""
	adj_list = dict()
	for variable in variables:
		adj_list[variable] = []
	edges =	[edge[0] for edge in sorted(
                    connection_tables.items(), key=lambda x: x[1], reverse=True)]
	np.random.shuffle(edges)
	for edge in edges:
		r = np.random.rand()
		if r <= connection_tables[edge]: 
			adj_list[edge[0]].append(edge[1])
			ebunch, nodes = adj_list_to_ebunch_and_nodes(adj_list)
			if not is_ebunch_dag(ebunch):
				adj_list[edge[0]].pop()
	# print(adj_list)
	return adj_list


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
	done = False
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
	Función para el pool de multiprocessiong 
	que genera un modelo a partir de las observaciones y calcula
	la función de probabilidad conjunta para una observación en 
	específico.
	"""
	pgmodel = generate_approx_model_from_graph(ebunch, nodes, df)
	modified_model.reset(pgmodel, ebunch, nodes)
	proba = modified_model.get_joint_prob_observation(nature_response)
	return (pair, proba)

def update_connection_beliefs(model, connection_tables, df, nature_response, use_causal_order=True):
	"""
	Función para actualizar las creencias usando 
	multiprocessing. Utiliza un modelo del agente acerca del mundo,
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
			if not use_causal_order and not is_ebunch_dag(ebunch_with_ij):
				continue
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
		# connection_tables[pair] = (connection_tables[pair] * p_sub_dict[pair]) / (p_sub_dict[pair] * connection_tables[pair] + p_complement_dict[pair] * (1 - connection_tables[pair]))
		if p_complement_dict.get(pair) != None and p_sub_dict.get(pair) != None:
			connection_tables[pair] = (connection_tables[pair] * p_sub_dict[pair]) / (p_sub_dict[pair] * connection_tables[pair] + p_complement_dict[pair] * (1 - connection_tables[pair]))
		# default_value = 0.00001
		# connection_tables[pair] = (connection_tables[pair] * p_sub_dict.get(pair, default_value)) / (p_sub_dict.get(pair, default_value) * connection_tables[pair] + p_complement_dict.get(pair, default_value) * (1 - connection_tables[pair]))
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
	for pair in connection_tables:
		cause = pair[0]
		effect = pair[1]
		if pair in ebunch:
			p_sub = p_obs_given_base_model
			ebunch_without_ij = copy(ebunch)
			ebunch_without_ij.remove((cause, effect))
			pgmodel_notij = generate_approx_model_from_graph(ebunch_without_ij, nodes, df)
			modified_model.reset(pgmodel_notij, ebunch_without_ij, nodes)
			p_complement = modified_model.get_joint_prob_observation(nature_response)
		else:
			ebunch_with_ij = copy(ebunch)
			ebunch_with_ij.append((cause, effect))	
			key_ebunch_with_ij = tuple(sorted(ebunch_with_ij))
			p_complement = p_obs_given_base_model
			pgmodel_ij = generate_approx_model_from_graph(ebunch_with_ij, nodes, df)
			modified_model.reset(pgmodel_ij, ebunch_with_ij, nodes)
			p_sub = modified_model.get_joint_prob_observation(nature_response)
		# print("Pij = {},  P~ij = {}".format(p_sub, p_complement))
		# print("{} * {} / ({} * {} + {} * (1 - {}))".format(connection_tables[pair], p_sub, p_sub, connection_tables[pair], p_complement, connection_tables[pair]))
		connection_tables[pair] = (connection_tables[pair] * p_sub) / (p_sub * connection_tables[pair] + p_complement * (1 - connection_tables[pair]))
	return connection_tables

def training(variables, rounds, connection_tables, data, unknown_model, nature, target, exploration_rate=1):
	intervention_vars = nature.model.get_intervention_variables()
	actions = {action : [0, 1] for action in intervention_vars}
	connection_probas = dict()
	update_prob_measures(connection_probas, connection_tables)
	local_data = deepcopy(data)
	df = pd.DataFrame.from_dict(local_data)
	pbar = tqdm.trange(rounds)
	rewards = [0]
	epsilon = 1.0
	for rnd in pbar:
		r = np.random.rand()
		# best action
		# epsilon = get_current_eps(epsilon, decay=0.9)
		# epsilon = get_current_eps_linear_decay(epsilon, rounds, rnd + 1)
		pbar.set_description(f"Training rounds epsilon = {epsilon} ")
		if r <= epsilon:
			idx_intervention_var = np.random.randint(len(intervention_vars))
			action = (intervention_vars[idx_intervention_var], np.random.randint(2))
			print(action)
		else:
			action = get_best_action(unknown_model, target, actions)
		# print(f"Action: {action}")
		nature_response = nature.action_simulator([action[0]], [action[1]])
		reward = nature_response[target["variable"]]
		rewards.append(reward)
		connection_tables = update_connection_beliefs(unknown_model, connection_tables, df, nature_response)
		update_prob_measures(connection_probas, connection_tables)
		for k in nature_response:
			local_data[k].append(nature_response[k])
		df = pd.DataFrame.from_dict(local_data)
		adj_list = create_graph_from_beliefs_unknown_order(variables, connection_tables)
		ebunch, nodes = adj_list_to_ebunch_and_nodes(adj_list)
		approx_model = generate_approx_model_from_graph(ebunch, nodes, df)
		unknown_model.reset(approx_model, ebunch, nodes)
	return connection_probas, rewards


def training_ligh_env_learning(variables, rounds, connection_tables, data_on, data_off, unknown_model_on, unknown_model_off, env, mod_episode=1):
	connection_probas = dict()
	local_data_on = deepcopy(data_on)
	local_data_off = deepcopy(data_off)
	df_on = pd.DataFrame.from_dict(local_data_on)
	df_off = pd.DataFrame.from_dict(local_data_off)
	update_prob_measures(connection_probas, connection_tables)
	pbar = tqdm.trange(rounds)
	rewards_per_block= []
	steps = 0
	policy = PolicyCM(linear=False)
	for rnd in pbar:
		pbar.set_description("Interaction rounds")
		done = False 
		env.reset()
		episode_reward = 0
		rewards_per_episode = []
		while not done:
			steps += 1
			targets = get_targets(env)
			action = policy.select_action(
				env, unknown_model_on, unknown_model_off)
			nature_response = action_simulator(env, action[0])
			done = nature_response.pop("done", None)
			reward = nature_response.pop("reward", None)
			episode_reward += reward
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
		rewards_per_episode.append(episode_reward)
		if rnd == 0 or (rnd + 1) % mod_episode == 0:
			rewards_per_block.append(np.mean(rewards_per_episode))
	for i in range(len(rewards_per_block)):
		print(i, rewards_per_block[i])
	return connection_probas, rewards_per_block


def light_env_learning(base_dir="results/light-switches", structure="one_to_one", num=5,
						rounds=50, l=1, experiments_per_structure=1, num_structures=10,
						causal_order=True):
	from env.light_env import LightEnv
	exploration_steps = l
	env = LightEnv(structure=structure, num=num)
	base_path = os.path.join(base_dir, structure, str(num))
	create_dirs_results(base_path)
	p_bar_structures = tqdm.trange(num_structures)
	start_time = time.time()
	for s in p_bar_structures:
		results_data = dict()
		p_bar_structures.set_description("Learning Structure")
		env.keep_struct = False
		env.reset()
		env.keep_struct = True
		lights_on_model = generate_model_from_env(env)
		lights_off_model = generate_model_from_env(env, lights_off=True)
		unknown_model_on = deepcopy(lights_on_model)
		unknown_model_off = deepcopy(lights_off_model)
		variables = sorted(lights_on_model.get_graph_toposort())
		causal_order = variables if causal_order else []
		invalid_edges = []
		causes = lights_on_model.get_intervention_variables()
		invalid_edges = generate_invalid_edges_light(variables, causes)
		global_beliefs_results = dict()
		rewards_per_struct = []
		base_structure_filename = f"light_env_struct_{structure}_{s}"
		lights_on_model.save_digraph_as_img(os.path.join(base_path, "graphs", base_structure_filename + ".pdf"))
		g_truth = {e : 1 for e in lights_on_model.digraph.edges}
		for i in range(experiments_per_structure):
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
			unknown_model_on.reset(approx_model_on, ebunch, nodes)
			unknown_model_off.reset(approx_model_off, ebunch, nodes)
			connection_probs, rewards = training_ligh_env_learning(
				variables, rounds, connection_tables, data_on, data_off, unknown_model_on, unknown_model_off, env)
			rewards_per_struct.append(rewards)
			for key in connection_probs:
				if key not in global_beliefs_results:
					global_beliefs_results[key] = []
				global_beliefs_results[key].append(connection_probs[key])
		results_data[f"gt_{s}"] = g_truth
		results_data[f"beliefs_{s}"] = global_beliefs_results
		results_data[f"training_time_{s}"] = time.time() - start_time
		results_data[f"rewards_{s}"] = rewards_per_struct
		dict_filename = os.path.join(base_path, "mats", base_structure_filename + ".pickle")
		with open(dict_filename, "wb") as handle:
			pickle.dump(results_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
		# labels = []
		# mean_vectors = []
		# std_dev_vectors = []
		# last_beliefs = dict()
		# for key in global_beliefs_results:
		# 	mean_vec = np.mean(global_beliefs_results[key], axis=0)
		# 	labels += [key]
		# 	mean_vectors.append(mean_vec)
		# 	std_dev_vectors.append(np.std(global_beliefs_results[key], axis=0))
		# 	last_beliefs[key] = mean_vectors[-1][-1]
		# x_axis = np.arange(len(mean_vectors[0]))
		# plot_measures(x_axis, mean_vectors, std_dev_vectors, labels, "{}/{}/all_lights_struct_{}_exp_{}_rounds_{}".format(base_dir, structure, s, experiments, rounds), legend=False)
def basic_model_learning(base_path="results/disease-treatment-best-action", experiments=10, rounds=50, plot_id=""):
	gt_ebunch = [("Reaction", "Lives"), ("Treatment", "Reaction"), ("Treatment", "Lives"), ("Disease", "Lives")]
	DG = nx.DiGraph([("Reaction", "Lives"), ("Treatment", "Reaction"), ("Treatment", "Lives"), ("Disease", "Lives")])
	# causal_order = list(nx.topological_sort(DG))
	causal_order = []
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
	g_truth = {e: 1 for e in DG.edges}
	n_exploration_steps = 1
	global_results = dict()
	for i in range(experiments):
		start_time = time.time()
		base_experiment_filename = f"disease-treatment-run-{i}-rounds-{rounds}"
		results_data = dict()
		local_exp_results = dict()
		data = explore_and_generate_data(nature, intervention_vars, n_steps=n_exploration_steps)
		df = pd.DataFrame.from_dict(data)
		connection_tables = create_pij(variables, causal_order, invalid_edges)
		adj_list = create_graph_from_beliefs_unknown_order(variables, connection_tables)
		ebunch, nodes = adj_list_to_ebunch_and_nodes(adj_list)
		approx_model = generate_approx_model_from_graph(ebunch, nodes, df)

		unknown_model = BaseModel('configs/incomplete_params.json')
		unknown_model.reset(approx_model, ebunch, nodes)
		unknown_model.show_graph()
		connection_probas, rewards = training(
			variables, rounds, connection_tables, data, unknown_model, nature, target)
		for key in connection_probas:
			if key not in global_results:
				global_results[key] = []
			global_results[key].append(connection_probas[key])
			local_exp_results[key] = connection_probas[key]
		results_data[f"gt_{i}"] = g_truth
		results_data[f"beliefs_{i}"] = local_exp_results
		results_data[f"training_time_{i}"] = time.time() - start_time
		results_data[f"rewards_{i}"] = rewards
		dict_filename = os.path.join(
			base_path, "mats", base_experiment_filename + ".pickle")
		with open(dict_filename, "wb") as handle:
			pickle.dump(results_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
	labels = []
	mean_vectors = []
	std_dev_vectors = []
	labels_correct_ones = []
	mean_vectors_correct_ones = []
	std_dev_vectors_correct_ones = []
	labels_wrong_ones = []
	mean_vectors_wrong_ones = []
	std_dev_vectors_wrong_ones = []
	for key in global_results:
		labels += [key]
		mean_vectors.append(np.mean(global_results[key], axis=0))
		std_dev_vectors.append(np.std(global_results[key], axis=0))
		if key in gt_ebunch:
			labels_correct_ones.append(key)
			mean_vectors_correct_ones.append(mean_vectors[-1])
			std_dev_vectors_correct_ones.append(std_dev_vectors[-1])
		else:
			labels_wrong_ones.append(key)
			mean_vectors_wrong_ones.append(mean_vectors[-1])
			std_dev_vectors_wrong_ones.append(std_dev_vectors[-1])
		print("{} {} {}".format(key, mean_vectors[-1][-1], std_dev_vectors[-1][-1]))
	x_axis = np.arange(len(mean_vectors[0]))
	plot_measures(x_axis, mean_vectors, std_dev_vectors, labels,
	              f"{base_path}/{plot_id}_connection_beliefs_exp_{experiments}_rounds_{rounds}_{intervention_vars}", outside_legend=True)
	plot_measures(x_axis, mean_vectors_correct_ones, std_dev_vectors_correct_ones, labels_correct_ones,
	              f"{base_path}/{plot_id}_connection_beliefs_correct_exp_{experiments}_rounds_{rounds}_{intervention_vars}", outside_legend=True)
	plot_measures(x_axis, mean_vectors_wrong_ones, std_dev_vectors_wrong_ones, labels_wrong_ones,
              f"{base_path}/{plot_id}_connection_beliefs_wrong_exp_{experiments}_rounds_{rounds}_{intervention_vars}", outside_legend=True)

	# for i in range(len(mean_vectors)):
	# 	plot_measures(x_axis, [mean_vectors[i]], [std_dev_vectors[i]], [labels[i]], "connection_beliefs_{}_exp_{}_rounds_{}_{}".format(labels[i], experiments, rounds, intervention_vars))
	

if __name__ == '__main__':
	for n in [5, 7, 9]:
		for struct in ["one_to_one", "one_to_many", "many_to_one"]:
			print(n, struct)
			light_env_learning(base_dir="results/light-switches-learning-and-using", structure=struct, num=n, rounds=500, num_structures=5)
			# break
		break
	# basic_model_learning(base_path="results/disease-treatment-random-action-several-actions", experiments=10, rounds=100, plot_id="shuffle")






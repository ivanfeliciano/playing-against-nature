#!/usr/bin/env python
# coding: utf-8

import itertools
from copy import deepcopy, copy
import logging

import numpy as np
import networkx as nx
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from scipy.stats import beta, dirichlet
import matplotlib.pyplot as plt
import pandas as pd


from model import BaseModel
from true_causal_model import TrueCausalModel
from agents.causal_agents import HalfBlindAgent
from utils.vis_utils import plot_measures
np.random.seed(0)


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
	return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def print_dict(data):
	for k in data:
		print("{} : {}".format(k, data[k]))

def plot_probabilities(connection_probas):
	for pair in connection_probas:
		plt.plot(connection_probas[pair], label=pair)
	plt.legend()
	plt.show()


def is_a_valid_edge(x, y, causal_order, invalid_edges):
	"""
	Verifica si un par ordenado (x, y) es una arista válida
	de acuerdo con el orden causal de las variables y si no
	es una arista inválida.
	"""
	if tuple(sorted((x, y))) in invalid_edges or causal_order.index(y) < causal_order.index(x):
		return False
	return True

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
		proba = np.random.rand()
		if is_a_valid_edge(pair[0], pair[1], causal_order, invalid_edges):
			connection_tables[(pair[0], pair[1])] = proba
		elif is_a_valid_edge(pair[1], pair[0], causal_order, invalid_edges):
			connection_tables[(pair[1], pair[0])] = proba
	return connection_tables

def create_graph_from_beliefs(variables, connection_tables):
	"""
	Retorna una gráfica de adyacencia a partir de la creencias
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
	nodes = []
	ebunch = []
	for node in adj_list:
		nodes.append(node)
		for child in adj_list[node]:
			ebunch.append((node, child))
	return ebunch, nodes

def explore_and_generate_data(nature, intervention_vars, n_steps=100):
	data_dict = dict()
	for i in range(n_steps):
		best_actions = np.random.randint(2, size=1)
		obs = nature.action_simulator(intervention_vars, best_actions)
		for k in obs:
			if k not in data_dict:
				data_dict[k] = []
			data_dict[k].append(obs[k])
	return data_dict

def update_prob_measures(connection_probas, connection_tables):
	for k in connection_tables:
		if k not in connection_probas:
			connection_probas[k] = []
		connection_probas[k] += [connection_tables[k]]

def generate_approx_model_from_graph(ebunch, nodes, df):
	df = df.sample(frac=1)
	approx_model = BayesianModel(ebunch)
	approx_model.add_nodes_from(nodes)
	approx_model.fit(df)
	return approx_model


def update_connection_beliefs(model, connection_tables, adj_list, df, nature_response):
	model_with_ij = deepcopy(model)
	model_without_ij = deepcopy(model)
	ebunch, nodes = adj_list_to_ebunch_and_nodes(adj_list)
	print("GRAPH")
	print("+" * 50)
	print_dict(adj_list)
	print(ebunch)
	print("+" * 50)
	for pair in connection_tables:
		cause = pair[0]
		effect = pair[1]
		print("Updating {} -> {}".format(cause, effect))
		if effect in adj_list[cause]:
			pgmodel_ij = generate_approx_model_from_graph(ebunch, nodes, df)
			model_with_ij.reset(pgmodel_ij, ebunch, nodes)
			print("Edge in graph")
			print("ebunch : {}".format(ebunch))

			ebunch_without_ij = copy(ebunch)
			ebunch_without_ij.remove((cause, effect))
			pgmodel_notij = generate_approx_model_from_graph(ebunch_without_ij, nodes, df)
			model_without_ij.reset(pgmodel_notij, ebunch_without_ij, nodes)
			print("Remove edge in graph")
			print("ebunch withoutij: {}".format(ebunch_without_ij))
		else:
			print("Edge not in graph")
			pgmodel_notij = generate_approx_model_from_graph(ebunch, nodes, df)
			model_without_ij.reset(pgmodel_notij, ebunch, nodes)
			print("ebunch withoutij : {}".format(ebunch))
			ebunch_with_ij = copy(ebunch)
			ebunch_with_ij.append((cause, effect))	
			print("Add edge in graph")
			print("ebunch with : {}".format(ebunch_with_ij))
			pgmodel_ij = generate_approx_model_from_graph(ebunch_with_ij, nodes, df)
			model_with_ij.reset(pgmodel_ij, ebunch_with_ij, nodes)
		
		# Calcula las probabilidades de observación dados los modelos
		p_sub = model_with_ij.get_joint_prob_observation(nature_response)
		p_complement = model_without_ij.get_joint_prob_observation(nature_response)
		print("Pij = {},  P~ij = {}".format(p_sub, p_complement))
		if isclose(p_sub, p_complement):
			print("-" * 50)
			print("MODEL WITH EDGE")
			for table in model_with_ij.pgmodel.get_cpds():
				print(table)
			print("-" * 50)
			print("-" * 50)
			print("MODEL WITHOUT EDGE")
			for table in model_without_ij.pgmodel.get_cpds():
				print(table)
			print("-" * 50)
		connection_tables[pair] = (connection_tables[pair] * p_sub) / (p_sub * connection_tables[pair] + p_complement * (1 - connection_tables[pair]))
	return connection_tables

def training(variables, rounds, agent, target, adj_list, connection_tables, data, unknown_model, nature):
	intervention_vars = agent.model.get_intervention_variables()
	connection_probas = dict()
	update_prob_measures(connection_probas, connection_tables)
	local_data = deepcopy(data)
	df = pd.DataFrame.from_dict(local_data)
	for rnd in range(rounds):
		print("*" * 50)
		print("Round {}".format(rnd))
		print_dict(connection_tables)
		best_actions = agent.make_decision(target, intervention_vars)
		nature_response = agent.nature.action_simulator(intervention_vars, best_actions)
		agent.rewards_per_round.append(nature_response[target["variable"]])
		connection_tables = update_connection_beliefs(unknown_model, connection_tables, adj_list, df, nature_response)
		update_prob_measures(connection_probas, connection_tables)
		
		for k in nature_response:
			local_data[k].append(nature_response[k])
		df = pd.DataFrame.from_dict(local_data)
		
		adj_list = create_graph_from_beliefs(variables, connection_tables)
		ebunch, nodes = adj_list_to_ebunch_and_nodes(adj_list)
		approx_model = generate_approx_model_from_graph(ebunch, nodes, df)
		unknown_model.reset(approx_model, ebunch, nodes)
		agent = HalfBlindAgent(nature, unknown_model)
		print("*" * 50)
	return connection_probas


def main():
	DG = nx.DiGraph([("Reaccion", "Final"), ("Tratamiento", "Reaccion"), ("Tratamiento", "Final"), ("Enfermedad", "Final")])
	causal_order = list(nx.topological_sort(DG))
	invalid_edges = sorted([("Enfermedad", "Tratamiento")])


	COMPLETE_MODEL = BaseModel('configs/model_parameters.json')
	nature = TrueCausalModel(COMPLETE_MODEL)
	variables = sorted(["Tratamiento", "Reaccion", "Enfermedad", "Final"])
	intervention_vars = COMPLETE_MODEL.get_intervention_variables()
	target_value = 1
	target = {
		"variable": COMPLETE_MODEL.get_target_variable(),
		"value" : target_value
	}

	experiments = 10
	global_results = dict()
	for i in range(experiments):
		rounds = 200
		data = explore_and_generate_data(nature, intervention_vars, n_steps=100)
		df = pd.DataFrame.from_dict(data)
		connection_tables = create_pij(variables, causal_order, invalid_edges)
		adj_list = create_graph_from_beliefs(variables, connection_tables)
		ebunch, nodes = adj_list_to_ebunch_and_nodes(adj_list)
		approx_model = generate_approx_model_from_graph(ebunch, nodes, df)

		unknown_model = BaseModel('configs/incomplete_params.json')
		unknown_model.reset(approx_model, ebunch, nodes)
		unknown_model.show_graph()
		agent = HalfBlindAgent(nature, unknown_model)

		connection_probas = training(variables, rounds, agent, target, adj_list, connection_tables, data, unknown_model, nature)
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
	x_axis = np.arange(len(mean_vectors[0]))
	plot_measures(x_axis, mean_vectors, std_dev_vectors, labels, "connection_beliefs_exp_{}_rounds_{}_{}".format(experiments, rounds, intervention_vars))
	for i in range(len(mean_vectors)):
		plot_measures(x_axis, [mean_vectors[i]], [std_dev_vectors[i]], [labels[i]], "connection_beliefs_{}_exp_{}_rounds_{}_{}".format(labels[i], experiments, rounds, intervention_vars))

if __name__ == '__main__':
	main()






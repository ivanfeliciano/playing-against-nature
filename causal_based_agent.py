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

from env.light_env import LightEnv
from utils.light_env_utils import *
from utils.helpers import *
from policy import PolicyCM
np.random.seed(0)

from structure_learning import *


def training(policy, variables, rounds, connection_tables, data_on, data_off, unknown_model_on, unknown_model_off, env, mod_episode=1):
	connection_probas = dict()
	local_data_on = deepcopy(data_on)
	local_data_off = deepcopy(data_off)
	df_on = pd.DataFrame.from_dict(local_data_on)
	df_off = pd.DataFrame.from_dict(local_data_off)
	update_prob_measures(connection_probas, connection_tables)
	pbar = tqdm.trange(rounds)
	actions = {f"cause_{i}": [1,] for i in range(env.num + 1)}
	rewards_per_block = []
	steps = 0
	ebunch = []
	for rnd in pbar:
		pbar.set_description("Interaction rounds")
		done = False 
		env.reset()
		episode_reward = 0
		rewards_per_episode = []
		while not done:
			steps += 1
			action = policy.select_action(env, ebunch, unknown_model_on, unknown_model_off)
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
	return connection_probas, rewards_per_block


def learning(base_dir="results/light-switches", structure="one_to_one", num=5,
						rounds=50, l=1, experiments_per_structure=1, num_structures=10, selection_scheme="dag_parents",
						linear=True):
	exploration_steps = l
	env = LightEnv(structure=structure, num=num)
	base_path = os.path.join(base_dir, structure, str(num))
	create_dirs_results(base_path)
	p_bar_structures = tqdm.trange(num_structures)
	start_time = time.time()
	for s in p_bar_structures:
		results_data = dict()
		policy = PolicyCM(linear=linear, selection_scheme=selection_scheme, nb_steps=rounds * env.num)
		p_bar_structures.set_description("Learning Structure")
		env.keep_struct = False
		env.reset()
		env.keep_struct = True
		lights_on_model = generate_model_from_env(env)
		lights_off_model = generate_model_from_env(env, lights_off=True)
		unknown_model_on = deepcopy(lights_on_model)
		unknown_model_off = deepcopy(lights_off_model)
		variables = sorted(lights_on_model.get_graph_toposort())
		causal_order = variables
		invalid_edges = []
		causes = lights_on_model.get_intervention_variables()
		invalid_edges = generate_invalid_edges_light(variables, causes)
		global_results = dict()
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
			connection_probs, rewards = training(
				policy, variables, rounds, connection_tables, data_on, data_off, unknown_model_on, unknown_model_off, env)
			for key in connection_probs:
				if key not in global_results:
					global_results[key] = []
				global_results[key].append(connection_probs[key])
		results_data[f"gt_{s}"] = g_truth
		results_data[f"beliefs_{s}"] = global_results
		results_data[f"training_time_{s}"] = time.time() - start_time
		results_data[f"rewards_{s}"] = rewards
		print(rewards)
		dict_filename = os.path.join(base_path, "mats", base_structure_filename + ".pickle")
		with open(dict_filename, "wb") as handle:
			pickle.dump(results_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def gt_baseline(base_dir="results/light-switches", structure="one_to_one", num=5, num_structures=1, rounds=100, experiments_per_structure=1):
	env = LightEnv(structure=structure, num=num)
	base_path = os.path.join(base_dir, structure, str(num))
	create_dirs_results(base_path)
	p_bar_structures = tqdm.trange(num_structures)
	for s in p_bar_structures:
		results_data = dict()
		p_bar_structures.set_description("Learning Structure")
		env.keep_struct = False
		env.reset()
		env.keep_struct = True
		aj_mat = env.aj
		aj_list, parents = aj_matrix_to_aj_list(aj_mat)
		base_structure_filename = f"light_env_struct_{structure}_{s}"
		for i in range(experiments_per_structure):
			pbar = tqdm.trange(rounds)
			rewards_per_episode = []
			for rnd in pbar:
				pbar.set_description("Interaction rounds")
				done = False 
				env.reset()
				episode_reward = 0
				while not done:
					action = get_good_action(env, parents)
					nature_response = action_simulator(env, action)
					done = nature_response.pop("done", None)
					reward = nature_response.pop("reward", None)
					episode_reward += reward
				rewards_per_episode.append(episode_reward)
		results_data[f"rewards_{s}"] = rewards_per_episode
		print(rewards_per_episode)
		dict_filename = os.path.join(base_path, "mats", base_structure_filename + ".pickle")
		with open(dict_filename, "wb") as handle:
			pickle.dump(results_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
	episodes = 50
	for n in [5, 7, 9]:
		for struct in ["one_to_one"]:#, "one_to_many", "many_to_one"]:
			print(n, struct)
			learning(base_dir="results/light-switches-using-params-linear-eps", structure=struct, l=1,
                            num_structures=1, rounds=episodes, num=n, selection_scheme="params",
							linear=True)
			learning(base_dir="results/light-switches-using-params-exp-decay", structure=struct, l=1,
                            num_structures=1, rounds=episodes, num=n, selection_scheme="params",
							linear=False)
			learning(base_dir="results/light-switches-using-dag-linear-eps", structure=struct, l=1,
                            num_structures=1, rounds=episodes, num=n, linear=True)
			learning(base_dir="results/light-switches-using-dag-exp-decay", structure=struct, l=1,
                            num_structures=1, rounds=episodes, num=n, linear=False)
			gt_baseline(base_dir="results/light-switches-gt", structure=struct, num=n, rounds=episodes)
			break
		break

import os
import math
import pickle

import numpy as np
import networkx as nx

from utils.vis_utils import *

def is_ebunch_dag(ebunch):
	G = nx.DiGraph()
	G.add_edges_from(ebunch)
	return nx.is_directed_acyclic_graph(G)

def powerset(n):
    powerset = []
    for i in range(1 << n):
        powerset.append(tuple([int(_) for _ in np.binary_repr(i, width=n)]))
    return powerset

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
	"""
	Checa si dos valores reales son iguales.
	"""
	return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def print_dict(data):
	for k in data:
		print("{} : {}".format(k, data[k]))


def is_a_valid_edge(x, y, causal_order, invalid_edges):
	"""
	Verifica si un par ordenado (x, y) es una arista válida
	de acuerdo con el orden causal de las variables y si no
	es una arista inválida.
	"""
	if (x, y) in invalid_edges or causal_order.index(y) < causal_order.index(x):
		return False
	return True

def compare_edges(g_truth, beliefs, epsilon=0.5):
	distance = 0.0
	for edge in beliefs:
		true_value = g_truth.get(edge, 0)
		pred = 0 if beliefs[edge] <= epsilon else beliefs[edge]
		distance += ((true_value - pred) ** 2 )
	return math.sqrt(distance)

def read_dict_from_pickle(pickle_filepath):
	"""
	Retorna un diccionario de python que fue 
	almacenado en un pickle.
	"""
	with open(pickle_filepath, "rb") as picke_file:
		return pickle.load(picke_file)

def metric_per_episode(gt, beliefs, operator):
	"""
	Calcula el promedio del xor o exactitud por episodio. De acuerdo
	con el valor de operator = {"xor", "equal"}
	"""
	edges = list(beliefs.keys())
	number_of_labels = len(edges)
	number_of_episodes = len(beliefs[edges[0]])
	results = []
	for i in range(number_of_episodes):
		current_sum = 0
		for edge in beliefs:
			val_gt = gt.get(edge, False)
			current_sum += (val_gt ^ beliefs[edge][i]) if \
                            operator == "xor" else (val_gt == beliefs[edge][i])
		results.append(current_sum)
	return np.array(results) / number_of_labels


def l2_loss(gt, beliefs):
	"""
	Calcula la función de pérdida l2.
	"""
	edges = list(beliefs.keys())

	number_of_episodes = len(np.squeeze(beliefs[edges[0]]))
	results = []
	for i in range(number_of_episodes):
		current_sum = 0
		for edge in beliefs:
			val_gt = gt.get(edge, 0)
			current_sum += (val_gt - np.squeeze(beliefs[edge])[i]) ** 2
		results.append(math.sqrt(current_sum))
	return results

def transform_to_boolean_values(dict_of_values, epsilon=0.7):
	"""
	docstring
	"""
	ans = dict()
	for pair in dict_of_values:
		values = np.squeeze([dict_of_values[pair]])
		if not values.shape:
			binary_values = bool(values)
		else:
			binary_values = list(map(lambda x: x > epsilon, values))
		ans[pair] = binary_values
	return ans

def apply_metric(gt, beliefs, metric_function, to_binary=False):
	"""
	Aplica la métrica de evaluación para comparar
	las aristas verdaderas y las creencias. Retorna un
	arreglo con las diferencias por episodio de entrenamiento.
	"""
	pass

def create_dirs_results(base_dir):
	"""
	Crea los directorios necesarios para guardar los 
	resultados de los experimentos:
		+ Plots : Se guardan las gráficas de pij
		+ Matrices de resultados : Archivos pickle que guardan las pij
		+ Grafos: Las estructuras del GT
	"""
	plots_path = os.path.join(base_dir, "plots")
	graphs_path = os.path.join(base_dir, "graphs")
	mats_path = os.path.join(base_dir, "mats")
	for p in [base_dir, plots_path, graphs_path, mats_path]:
		if not os.path.exists(p):
			os.makedirs(p)

def get_current_eps(epsilon, decay=0.9, min_eps=0.01):
	"""
	Calcula un nuevo valor para la tasa de exploración.
	El decremento es exponencial.
	"""
	return max(epsilon * decay, min_eps)

def get_current_eps_linear_decay(epsilon, n_steps, step, min_eps=0.01, max_eps=1.0):
	a = -float(max_eps - min_eps) / float(n_steps)
	b = float(max_eps)
	return max(min_eps, a * float(step) + b)

def get_struct_index(data):
	bel_key = ""
	for k in data:
		if k.startswith("beliefs"):
			bel_key = k
			break
	gt_key = f"gt_{bel_key.strip().split('_')[1]}"
	return bel_key, gt_key
def beliefs_to_mat(data, n, timestep=-1):
	beliefs_k, gt_k = get_struct_index(data)
	gt = data[gt_k]
	beliefs = data[beliefs_k]
	gt_mat = np.zeros((2 * n + 1, 2 * n + 1))
	beliefs_mat = np.zeros((2 * n + 1, 2 * n + 1))
	for pair in gt:
		i = int(pair[0].strip().split("_")[1])
		j = int(pair[1].strip().split("_")[1])
		gt_mat[i][n + 1 + j] = gt[pair]
	for pair in beliefs:
		values = np.squeeze(beliefs[pair])
		cause_key = pair[0].strip().split("_")[0]
		effect_key = pair[1].strip().split("_")[0]
		if cause_key == "cause":
			i = int(pair[0].strip().split("_")[1])
		else:
			i = int(pair[0].strip().split("_")[1]) + n + 1
		if effect_key == "cause":
			j = int(pair[1].strip().split("_")[1])
		else:
			j = int(pair[1].strip().split("_")[1]) + n + 1
		beliefs_mat[i][j] = values[timestep]
	return gt_mat, beliefs_mat
if __name__ == "__main__":
	path = "/home/ivan/Documentos/playing-against-nature/results/light-switches/many_to_one/5/mats/light_env_struct_many_to_one_2.pickle"
	data = read_dict_from_pickle(path)
	gt, beliefs = beliefs_to_mat(data, 5)
	plot_heatmap(gt, "gt_heatmap")
	plot_heatmap(beliefs, "beliefs_heatmap")

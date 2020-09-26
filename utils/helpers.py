import os
import math

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
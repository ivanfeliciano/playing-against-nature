import numpy as np
import networkx as nx
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import json


class BaseModel(object):
	"""
	Un objeto de este tipo contiene al modelo gráfico probabilista, incluye su grafo
	y sus parámetros (CPD) además de un objeto para hacer inferencia.

	Args:
		config_file_path (str) : la ruta al json con la información de
		DAG y sus tablas de probabilidad condicional. 
	"""
	def __init__(self, config_file_path):
		self.config_file_path = config_file_path
		self.digraph = None
		self.pgmodel = None
		self.infer_system = None
		with open(config_file_path) as json_file:
			data = json.load(json_file)
		self.init_graph(data['digraph'])
		self.init_model(data['digraph'], data['cpdtables'])
		self.targets = data['targets']
		self.nature_variables = data['nature_variables']
	def init_graph(self, ebunch, plot=False, graph_id='dag'):
		"""
		Creo el DAG con DiGraph de la biblioteca networkx usando
		una lista de aristas.

		Args:
			ebunch (list) : una lista de que contiene a las aristas del grafo.
			plot (boolean) : una bandera para saber si guardo una imagen del grafo
			usando matplotlib.
			graph_id (str): el nombre para identificar el grafo. 
		"""
		self.digraph =  nx.DiGraph(ebunch)
		if plot: self.save_digraph_as_img(graph_id)
	def init_model(self, ebunch, cpdtables, plot=False, pgm_id='pgm'):
		"""
		Creo el PGM usando PGMPY. Por ahora es un modelo Bayesiano. Recibe 
		la listas de aristas y las tablas CPD.

		Args:
			ebunch (list) : una lista de que contiene a las aristas del grafo.
			cpdtables (list) : un arreglo de diccionarios donde cada diccionario 
			contiene la información necesaria para crear una tabla de probabilidad.
			plot (boolean) : una bandera para saber si guardo una imagen del grafo
			usando matplotlib.
			graph_id (str): el nombre para identificar el grafo. 
		"""
		self.pgmodel = BayesianModel(ebunch)
		for cpdtable in cpdtables:
			cpdtable = TabularCPD(variable=cpdtable['variable'],\
						variable_card=cpdtable['variable_card'],\
						values=cpdtable['values'],\
						evidence_card=cpdtable.get('evidence_card'),\
						evidence=cpdtable.get('evidence'))
			self.pgmodel.add_cpds(cpdtable)
		if not self.pgmodel.check_model():
			raise ValueError("Error with CPDTs")
		self.infer_system = VariableElimination(self.pgmodel)
		if plot: self.save_pgm_as_img(pgm_id)
	def get_nature_variables(self):
		"""
		Regresa una lista con las variables que la naturaleza mueve.
		"""
		return self.nature_variables
	def get_nature_var_prob(self, nature_variable):
		"""
		Regresa una lista con las probabilidades de los valores
		de una variable de la naturaleza dada como argumento.

		Args:
			nature_variable (str) : nombre de la variable.
		"""
		if nature_variable in self.nature_variables:
			return np.squeeze(self.pgmodel.get_cpds(nature_variable).get_values())
	def make_inference(self, variable, evidence):
		"""
		Ejecuta el motor de inferencia para obtener el valor de una variable
		dada la evidencia en un diccionario.

		Args:
			variable (str) : nombre de la variable a inferir.
			evidence (dict) : un diccionario con la evidencia de otras variables de la forma {variable :  value}.
		"""
		return self.infer_system.map_query([variable],\
			evidence=evidence)[variable]
	def save_digraph_as_img(self, filename):
		"""
		Método auxiliar para guardar el DAG de networkx como imagen.
		"""
		nx.draw(self.digraph, with_labels=True)
		plt.savefig(filename)
		plt.clf()
	def save_pgm_as_img(self, filename):
		"""
		Método auxiliar para guardar el DAG del pgmpy como imagen.
		"""
		nx.draw(self.digraph, with_labels=True)
		plt.savefig(filename)
		plt.clf()
	def get_graph_toposort(self):
		"""
		Método que regresa una lista con las variables en orden topológico
		del DAG.
		"""
		return list(nx.topological_sort(self.digraph))

def main():
	test_model = BaseModel('model_parameters.json')
	# for cpdt in test_model.pgmodel.get_cpds():
		# print(cpdt)
	# print(test_model.get_graph_toposort())
if __name__ == '__main__':
	main()
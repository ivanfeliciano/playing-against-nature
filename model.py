import logging
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
		
		data (dict) : si no se cuenta con un archivo de configuración
		se puede utilizar un diccionario con los elementos para inicializar
		el objeto.

	to-do : por ahora sólo funciona con valores binarias. 
	"""
	def __init__(self, config_file_path=None, data=None):
		self.config_file_path = config_file_path
		self.digraph = None
		self.pgmodel = None
		self.infer_system = None
		self.ebunch = None
		self.nodes = None
		self.variables_dict = dict()
		if config_file_path:
			with open(config_file_path) as json_file:
				data = json.load(json_file)
		if data.get('digraph'):
			self.ebunch = data['digraph']
			self.pgmodel = BayesianModel(self.ebunch)
			self.nodes = data.get('nodes', [])
			if self.nodes:
				self.pgmodel.add_nodes_from(self.nodes)
			self.init_graph(ebunch=self.ebunch, nodes=self.nodes)
		if data.get('cpdtables'):
			self.init_model(self.ebunch, data['cpdtables'])
			for table in self.pgmodel.get_cpds():
				logging.info(table)
		self.target = data['target']
		self.nature_variables = data['nature_variables']
		self.intervention_variables = data['interventions']
	def init_graph(self, ebunch, nodes=[], plot=True, graph_id='figures/dag'):
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
		for node in nodes:
			self.digraph.add_node(node)
		if plot: self.save_digraph_as_img(graph_id)
	def reset(self, pgmodel, ebunch, nodes=[]):
		"""
		Método para cambiar el modelo y el grafo. Además,
		se actualiza el sistema de inferencia de acuerdo con el nuevo
		modelo. Este método se utiliza para hacer un modelo dinámico 
		donde lo único que se mantienen son las variables.
		"""
		self.init_graph(ebunch, nodes=nodes, plot=False)
		for variable in pgmodel.nodes():
			self.variables_dict[variable] = [0, 1]
		self.ebunch = ebunch
		self.nodes = nodes
		self.pgmodel = pgmodel
		self.update_infer_system()

	def show_graph(self):
		"""
		Usa matplolib para mostrar el grafo causal del modelo.
		"""
		pos = nx.circular_layout(self.digraph)
		nx.draw(self.digraph, with_labels=True, pos=pos)
		plt.show()
		plt.clf()
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
		for cpdtable in cpdtables:
			self.variables_dict[cpdtable['variable']] = [\
				_ for _ in range(cpdtable['variable_card'])]
			table = TabularCPD(variable=cpdtable['variable'],\
						variable_card=cpdtable['variable_card'],\
						values=cpdtable['values'],\
						evidence_card=cpdtable.get('evidence_card'),\
						evidence=cpdtable.get('evidence'))
			if cpdtable.get('evidence'):
				table.reorder_parents(sorted(cpdtable.get('evidence')))
			self.pgmodel.add_cpds(table)
		if not self.pgmodel.check_model():
			raise ValueError("Error with CPDTs")
		self.update_infer_system()
		if plot: self.save_pgm_as_img(pgm_id)
	def update_infer_system(self):
		"""
		Actualiza el sistema de inferencia para que sea compatible con
		el pgm. Usa VariableElimination.
		"""
		self.infer_system = VariableElimination(self.pgmodel)

	def get_variable_values(self, variable):
		"""
		Obtiene una lista de los valores que puede
		tomar una variable.
		"""
		return self.variables_dict.get(variable)
	def get_target_variable(self):
		"""
		Regresa una lista con las variables objetivo.
		"""
		return self.target
	def get_intervention_variables(self):
		"""
		Regresa una lista con las variables intervenibles.
		"""
		return self.intervention_variables
	def get_nature_variables(self):
		"""
		Regresa una lista con las variables que la naturaleza mueve.
		"""
		return self.nature_variables
	def get_ebunch(self):
		"""
		Regresa lista de aristas del modelo.
		"""
		return self.ebunch

	def get_nodes(self):
		"""
		Regresa lista de nodos aislados del modelo.
		"""
		return self.nodes
	def get_nature_var_prob(self, nature_variable):
		"""
		Regresa una lista con las probabilidades de los valores
		de una variable de la naturaleza dada como argumento.

		Args:
			nature_variable (str) : nombre de la variable.
		"""
		if nature_variable in self.nature_variables:
			return np.squeeze(self.pgmodel.get_cpds(nature_variable).get_values())
	def conditional_probability(self, variable, evidence):
		"""
		Calcula la probabilidad de todos los valores de una variable
		dada la evidencia usando el método de eliminación de 
		variable.
		"""
		return self.infer_system.query([variable], \
			evidence=evidence, show_progress=False)
	def make_inference(self, variable, evidence):
		"""
		Ejecuta el motor de inferencia para obtener el valor de una variable
		dada la evidencia en un diccionario.

		Args:
			variable (str) : nombre de la variable a inferir.
			evidence (dict) : un diccionario con la evidencia de otras variables de la forma {variable :  value}.
		"""
		return self.infer_system.map_query([variable],\
                                     evidence=evidence, show_progress=False)[variable]
	def save_digraph_as_img(self, filename):
		"""
		Método auxiliar para guardar el DAG de networkx como imagen.
		"""
		pos = nx.circular_layout(self.digraph)
		nx.draw(self.digraph, with_labels=True, pos=pos)
		plt.savefig(filename)
		plt.show()
		plt.clf()
	def save_pgm_as_img(self, filename):
		"""
		Método auxiliar para guardar el DAG del pgmpy como imagen.
		"""
		nx.draw(self.digraph, with_labels=True)
		plt.show()
		plt.savefig(filename)
		plt.clf()
	def get_graph_toposort(self):
		"""
		Método que regresa una lista con las variables en orden topológico
		del DAG.
		"""
		return list(nx.topological_sort(self.digraph))
	def get_nodes_and_predecessors(self):
		"""
		Regresa un arreglo de duplas nodo, predecesores ordenados.
		"""
		return { node : sorted(self.digraph.predecessors(node)) \
			for node in self.digraph.nodes
		}
	def get_number_of_values(self, variable):
		"""
		to-do : un método para que me regrese cuantos valores posibles tiene
		una variable y tal vez hasta los valores correspondientes
		"""
		return len(self.variables_dict.get(variable, []))
	def get_joint_prob_observation(self, observation):
		"""
		Obtiene la probabilidad de una observación.
		"""
		prob = self.infer_system.query(variables=list(observation.keys()), joint=True, show_progress=False)
		variables = prob.variables
		values = prob.values
		for i in range(len(variables)):
			value = observation[variables[i]]
			values = values[value]
		return values

def main():
	test_model = BaseModel('model_parameters.json')
	# for cpdt in test_model.pgmodel.get_cpds():
		# print(cpdt)
	# print(test_model.get_graph_toposort())
if __name__ == '__main__':
	main()

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
		self.digraph =  nx.DiGraph(ebunch)
		if plot: self.save_digraph_as_img(graph_id)
	def init_model(self, ebunch, cpdtables, plot=False, pgm_id='pgm'):
		self.pgmodel = BayesianModel(ebunch)
		for cpdtable in cpdtables:
			cpdtable = TabularCPD(variable=cpdtable['variable'],\
						variable_card=cpdtable['variable_card'],\
						values=cpdtable['values'],\
						evidence_card=cpdtable.get('evidence_card'),\
						evidence=cpdtable.get('evidence'))
			self.pgmodel.add_cpds(cpdtable)
		if not self.pgmodel.check_model():
			raise ValueError("Error with CPTS")
		self.infer_system = VariableElimination(self.pgmodel)
		if plot: self.save_pgm_as_img(pgm_id)
	def get_nature_variables(self):
		return self.nature_variables
	def get_nature_var_prob(self, nature_variable):
		if nature_variable in self.nature_variables:
			return np.squeeze(self.pgmodel.get_cpds(nature_variable).get_values())
	def make_inference(self, variable, evidence):
		return self.infer_system.map_query([variable],\
			evidence=evidence)[variable]
	def save_digraph_as_img(self, filename):
		nx.draw(self.digraph, with_labels=True)
		plt.savefig(filename)
		plt.clf()
	def save_pgm_as_img(self, filename):
		nx.draw(self.digraph, with_labels=True)
		plt.savefig(filename)
		plt.clf()
	def get_graph_toposort(self):
		return list(nx.topological_sort(self.digraph))

def main():
	test_model = BaseModel('model_parameters.json')
	# for cpdt in test_model.pgmodel.get_cpds():
		# print(cpdt)
	# print(test_model.get_graph_toposort())
if __name__ == '__main__':
	main()
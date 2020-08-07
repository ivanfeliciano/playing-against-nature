import itertools
from copy import deepcopy, copy

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

np.random.seed(0)

COMPLETE_MODEL = BaseModel('configs/model_parameters.json')
nature = TrueCausalModel(COMPLETE_MODEL)

intervention_vars = ["Tratamiento"]
target_value = 1
target = {
    "variable": "Final",
    "value" : target_value
}

data = dict()
for i in range(100):
	best_actions = np.random.randint(2, size=len(intervention_vars))
	obs = nature.action_simulator(intervention_vars, best_actions)
	for k in obs:
		if k not in data:
			data[k] = []
		data[k].append(obs[k])
df = pd.DataFrame.from_dict(data)
print("Termina exploración")
print(df)

model = BayesianModel([('Reaccion', 'Final'),\
						('Tratamiento', 'Final')])
print("Inicia interacción")

model.fit(df)
for cpd in model.get_cpds():
	print(cpd)

for i in range(100):
	best_actions = np.random.randint(2, size=len(intervention_vars))
	obs = nature.action_simulator(intervention_vars, best_actions)
	for k in obs:
		data[k].append(obs[k])

df = pd.DataFrame.from_dict(data)

model = BayesianModel([('Reaccion', 'Final'),\
						('Tratamiento', 'Final'),\
						('Tratamiento', 'Reaccion'),\
						('Enfermedad', 'Final')])
model.fit(df)
print("***************")
for cpd in model.get_cpds():
	print(cpd)

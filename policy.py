import random

import numpy as np

from utils.light_env_utils import get_targets, parents_from_ebunch

def do_calculus(infer_system, variable, evidence):
	"""
	Calcula la tabla de probabilidad condicional para una
	variable efecto dada una variable de acción.
	"""
	return infer_system.query([variable],
                    evidence=evidence, show_progress=False).values

def get_best_action(model, target, actions):
	"""
	Selecciona la mejor acción a realizar dada el modelo actual
	y la variable objetivo.
	"""
	best_action = None
	max_prob = -float("inf")
	for action in actions:
		for val in actions[action]:
			query_dict = { action : val }
			prob_table = do_calculus(model.infer_system, target["variable"], query_dict)
			prob = prob_table[target["value"]]
			if max_prob < prob:
				max_prob = prob
				best_action = (action, val)
	return best_action

class Policy(object):
	def __init__(self, nb_steps=1, linear=True, eps_max=1.0, eps_min=0.01, selection_scheme="params"):
		self.eps_max = eps_max
		self.eps_min = eps_min
		self.nb_steps = nb_steps
		self.step = 0
		self.current_eps = eps_max
		self.select_scheme = selection_scheme
		self.linear = linear

	def get_current_value(self, decay=0.99):
		if not self.linear:
			self.current_eps *= decay
			return max(self.current_eps, self.eps_min)
		a = -float(self.eps_max - self.eps_min) / float(self.nb_steps)
		value = max(self.eps_min, a * float(self.step) + self.eps_max)
		self.step += 1
		self.current_eps = value
		return value
	def select_action(self, env, state, Q):
		r = np.random.uniform()
		eps = self.get_current_value()
		if r < eps:
			return env.action_space.sample()
		return np.argmax(Q[state])


class PolicyCM(Policy):
	def select_action(self, env, unknown_model_on, unknown_model_off, ebunch=None):
		actions = {f"cause_{i}": [1, ] for i in range(env.num + 1)}
		targets = []
		targets = get_targets(env)
		r = np.random.uniform()
		eps = self.get_current_value()
		action = action = (f"cause_{env.num}",)
		if r > eps:
			if len(targets):
				target_name = random.choice(list(targets.keys()))
				target_value = targets[target_name]
				target = dict(variable=target_name, value=target_value)
				if self.select_scheme == "dag_parents":
					action = self.select_action_using_dag(target, ebunch, env)
				else:
					action = self.select_action_using_params(actions, target,\
						unknown_model_on, unknown_model_off)
			return action
		action_idx = env.action_space.sample()
		return (f"cause_{action_idx}",)
	def select_action_using_dag(self, target, ebunch, env):
		parents = parents_from_ebunch(ebunch)
		if target["variable"] in parents and len(target["variable"]) > 0:
			return (random.choice(parents[target["variable"]]),)
		return None
	def select_action_using_params(self, actions, target, unknown_model_on, unknown_model_off):
		if target["value"] == 1:
			return get_best_action(unknown_model_on, target, actions)
		else:
			return get_best_action(unknown_model_off, target, actions)


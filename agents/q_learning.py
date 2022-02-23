import pickle
import logging
import numpy as np

from agents.core import Agent
from utils.helpers import powerset, get_current_eps, get_current_eps_linear_decay
from utils.light_env_utils import obs_to_tuple
from env.light_env import LightEnv

class Policy(object):
	def __init__(self, eps_max, eps_min, eps_test, nb_steps):
		self.eps_max = eps_max
		self.eps_min = eps_min
		self.eps_test = eps_test
		self.nb_steps = nb_steps
		self.step = 0
		self.current_eps = eps_max

	def get_current_value(self, training=True):
		if training:
			a = -float(self.eps_max - self.eps_min) / float(self.nb_steps)
			b = float(self.eps_max)
			value = max(self.eps_min, a * float(self.step) + b)
		else:
			value = self.eps_test
		self.step += 1
		self.current_eps = value
		return value
	def select_action(self, state, Q, training=True):
		raise NotImplementedError

class EpsilonGreedy(Policy):
	def select_action(self, state, Q, env, training=True, prob_explore=0.0):
		r = np.random.uniform()
		eps = self.get_current_value(training)
		if r > eps:
			return np.argmax(Q[state])
		return env.action_space.sample()

class QLearningLightsSwitches(object):
	"""
	Clase base para un agente de Q learning.
	Aquí se configura el ambiente, los parámetros y
	el flujo del aprendizaje.
	"""
	def __init__(self, environment, policy, episodes=100, alpha=0.8, gamma=0.95, mod_episode=1):
		self.env = environment
		self.policy = policy
		self.episodes = episodes
		self.alpha = alpha
		self.gamma = gamma
		self.Q = self.init_q_table()
		self.mod_episode = mod_episode
		self.training = True
	def select_action(self, state):
		if self.training:
			return self.policy.select_action(state, self.Q, env)
		return self.policy.select_action(state, self.Q, training=False)
	def train(self):
		self.training = True
		self.avg_reward = []
		rewards_per_episode = []
		for episode in range(self.episodes):
			total_episode_reward = 0
			state = self.env.reset()
			state = obs_to_tuple(state, self.env.num)
			done = False
			assisted_times = 0
			while not done:
				action = self.select_action(state)
				new_state, reward, done, info = self.env.step(action)
				new_state = obs_to_tuple(new_state, self.env.num)
				self.Q[state][action] = self.Q[state][action] + self.alpha * \
										(reward + self.gamma * np.max(self.Q[new_state]) -\
										self.Q[state][action])
				state = new_state
				total_episode_reward += reward
			rewards_per_episode.append(total_episode_reward)
			if episode == 0 or (episode + 1) % self.mod_episode == 0:
				rewards_per_episode = self.update_avg_reward(rewards_per_episode)
		return self.avg_reward
	def update_avg_reward(self, rewards_per_episode):
		episodes_block_avg_rwd = np.mean(rewards_per_episode)
		self.avg_reward.append(episodes_block_avg_rwd)
		return []
	def init_q_table(self):
		all_states = powerset(self.env.num)
		return {
			state :  np.zeros(self.env.num + 1) for state in all_states
		}

class QLearning(Agent):
	"""
	Q learning para un bandido con una sola acción.
	"""
	def __init__(self, nature, epsilon=1.0):
		super().__init__(nature)
		self.epsilon = epsilon
		self.action_variable = self.nature.model.get_intervention_variables()
		self.action_values = self.nature.model.get_variable_values(self.action_variable[0])
		self.target = self.nature.model.get_target_variable()
		self.k = np.zeros(len(self.action_values), dtype=np.int)
		self.Q = np.zeros(len(self.action_values), dtype=np.float)
	
	def update_q(self, action, reward):
		self.k[action] += 1
		self.Q[action] += (1. / self.k[action]) * (reward - self.Q[action])
	
	def make_decision(self, i=0, rounds=0):
		r = np.random.random()
		self.epsilon = get_current_eps_linear_decay(self.epsilon, rounds, i)
		if r < self.epsilon:
			return np.random.randint(len(self.action_values))
		else:
			action = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
		return action

	def training(self, rounds):
		for i in range(rounds):
			action = np.random.randint(len(self.action_values))
			# action = self.make_decision(i + 1, rounds)
			state = self.nature.action_simulator(self.action_variable, [action])
			reward = state[self.target]
			self.update_q(action, reward)
			self.rewards_per_round.append(reward)
		logging.info(self.rewards_per_round)
		return self.rewards_per_round


if __name__ == "__main__":
	from true_causal_model import TrueCausalModel
	from model import BaseModel
	logging.basicConfig(filename='logs/qlearningAgent.log',
						filemode='w', level=logging.INFO)
	model = BaseModel('configs/model_parameters.json')
	nature = TrueCausalModel(model)
	for cpd in nature.model.pgmodel.get_cpds():
		print(cpd)
	rounds = 50
	for i in range(10):
		dict_filename = f"results/random-disease/mats/random-disease_{i}"
		rewards_dict = dict()
		qlearning_agent = QLearning(nature)
		qlearning_agent.training(rounds)
		rewards_dict[f"rewards_{i}"] = qlearning_agent.rewards_per_round
		with open(dict_filename, "wb") as handle:
			pickle.dump(rewards_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
	# env = LightEnv()
	# env.keep_struct = False
	# env.reset()
	# env.keep_struct = True
	# episodes = 100
	# eps_policy = EpsilonGreedy(1, 0.1, 0.1, 1)
	# vanilla_q_learning = QLearningLightsSwitches(env, eps_policy, episodes=episodes, mod_episode=5)
	# vanilla_q_learning.train()
	# for i in range(len(vanilla_q_learning.avg_reward)):
	# 	print(i, vanilla_q_learning.avg_reward[i])
	
	

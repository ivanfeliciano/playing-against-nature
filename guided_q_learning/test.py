from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from env.light_env import LightEnv
from environments import LightAndSwitchEnv
from agents import QLearningAgent
from utils.lights_env_helper import aj_to_adj_list, remove_edges, to_wrong_graph
from policy import EpsilonGreedy

episodes = 50000
struct = "one_to_one"
num = 5
env = LightEnv(horizon=num, num=num, structure=struct)
env.keep_struct = False
env.reset()
env.keep_struct = True
full_adj_list = aj_to_adj_list(env.aj)
incomplete_adj_list = remove_edges(deepcopy(full_adj_list))
incorrect_adj_list = to_wrong_graph(deepcopy(full_adj_list), n_effects=num)

vanilla_env = LightAndSwitchEnv(deepcopy(env), full_adj_list)
full_info_environment = LightAndSwitchEnv(deepcopy(env), full_adj_list)
partial_info_environment = LightAndSwitchEnv(deepcopy(env), incomplete_adj_list)
wrong_info_environment = LightAndSwitchEnv(deepcopy(env), incorrect_adj_list)

full_info_environment.causal_structure.draw_graph("full_structure")
partial_info_environment.causal_structure.draw_graph("partial_structure")
wrong_info_environment.causal_structure.draw_graph("wrong_structure")

eps_policy = EpsilonGreedy(1, 0.1, 0.1, num + 1, full_info_environment)
causal_eps_policy = EpsilonGreedy(1, 0.1, 0.1, num + 1, full_info_environment, True)
partial_causal_eps_policy = EpsilonGreedy(1, 0.1, 0.1, num + 1, partial_info_environment, True)
wrong_causal_eps_policy = EpsilonGreedy(1, 0.1, 0.1, num + 1, wrong_info_environment, True)

vanilla_q_learning = QLearningAgent(vanilla_env, eps_policy, episodes=episodes, mod_episode=100)
causal_q_learning = QLearningAgent(full_info_environment, causal_eps_policy, episodes=episodes, mod_episode=100)
partial_causal_q_learning = QLearningAgent(partial_info_environment, partial_causal_eps_policy, episodes=episodes, mod_episode=100)
wrong_causal_q_learning = QLearningAgent(wrong_info_environment, wrong_causal_eps_policy, episodes=episodes, mod_episode=100)

a = vanilla_q_learning.train()
b = causal_q_learning.train()
c = partial_causal_q_learning.train()
d = wrong_causal_q_learning.train()

x_axis = 100 * (np.arange(len(a)))
        
plt.plot(x_axis, a, label="Q-learning", color="#2980b9")
plt.plot(x_axis, b, label="Assisted Q-learning", color="#CC4F1B")
plt.plot(x_axis, c, label="Partially assisted Q-learning", color="#138d75")
plt.plot(x_axis, d, label="Wrond assisted Q-learning", color="#f39c12")
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.legend()
plt.savefig("plots/comparison.png")  
plt.close()

vanilla_q_learning.plot_avg_reward("vanilla_q_learning")
causal_q_learning.plot_avg_reward("causal_q_learning")
partial_causal_q_learning.plot_avg_reward("partial_causal_q_learning")
wrong_causal_q_learning.plot_avg_reward("wrong_causal_q_learning")

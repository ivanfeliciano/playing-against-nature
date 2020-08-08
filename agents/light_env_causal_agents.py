from env.light_env import LightEnv
import logging
import json
import itertools
from copy import deepcopy

import numpy as np
import networkx as nx
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from scipy.stats import beta, dirichlet
import matplotlib.pyplot as plt

from agents.causal_agents import CausalAgent, HalfBlindAgent
from utils.light_env_utils import check_diff_and_get_target_variables

class LightsFullyInformedAgent(CausalAgent): 
    def training(self, env, episodes):
        intervention_vars = self.model.get_intervention_variables()
        horizon = env.horizon
        for i in range(episodes):
            done = False
            self.nature.reset()
            episode_reward = 0
            k = 0
            while k < horizon:
                k += 1
                target_variables, targets_values = check_diff_and_get_target_variables(self.nature.env)
                if len(target_variables) < 1:
                    break
                idx = np.random.randint(len(target_variables))
                target = {
                    "variable": target_variables[idx],
                 	"value": targets_values[idx]
                     }
                best_actions = self.make_decision(target, intervention_vars)
                logging.info("Best actions {} {}".format(intervention_vars, best_actions))
                for action_idx in range(len(best_actions)):
                    if best_actions[action_idx] == 1:
                        nature_response = self.nature.action_simulator(\
                                                intervention_vars[action_idx])
                        episode_reward += nature_response["reward"]
                        logging.info(nature_response)
                        done = self.nature.done
                self.rewards_per_round.append(episode_reward)
        return self.rewards_per_round

class LightsHalfBlindAgent(object):
    pass


if __name__ == "__main__":
    from utils.light_env_utils import generate_model_from_env
    from utils.vis_utils import plot_reward
    from true_causal_model import TrueCausalModelEnv
    env = LightEnv(structure="one_to_one")
    env.keep_struct = False
    env.reset()
    env.keep_struct = True
    model = generate_model_from_env(env)
    nature_light_switch = TrueCausalModelEnv(env, model)
    episodes = 5
    agent = LightsFullyInformedAgent(nature=nature_light_switch, pgmodel=model)
    rewards = agent.training(env, episodes)
    print(rewards)
    std_dev = np.zeros(len(rewards))
    plot_reward(rewards, std_dev, "FullyInformedAgent", "Lights")

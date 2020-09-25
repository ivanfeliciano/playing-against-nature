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
        """
        El problema puede resolverse haciendo de a intervenciones atómicas
        """
        intervention_vars = self.model.get_intervention_variables()
        horizon = env.horizon
        for i in range(episodes):
            print("Episode : {}".format(i))
            done = False
            self.nature.reset()
            env.reset()
            episode_reward = 0
            k = 0
            nature_response = dict()
            step = 0
            while not done:
                k += 1
                print("Current state : {}".format(env._get_obs()[:env.num]))
                print("Goal : {}".format(env.goal))
                target_variables, targets_values = check_diff_and_get_target_variables(env)
                print(target_variables, targets_values)
                if len(target_variables) < 1:
                    break
                idx = np.random.randint(len(target_variables))
                target = {
                    "variable": target_variables[idx],
                 	"value": targets_values[idx]
                    }
                best_actions = []
                print("Light {} should be changet to {}".format(target["variable"], target["value"]))
                for intervention_var in intervention_vars:
                    a = self.make_decision_advanced(target, [intervention_var],threshold=0.6)
                    best_actions.append(a)
                logging.info("Best actions {} {}".format(intervention_vars, best_actions))
                for action_idx in range(len(best_actions)):
                    if best_actions[action_idx] != None and not done:
                        step += 1
                        print("Step {}".format(step))
                        print("Best action to change {} is to act on swith {}".format(target["variable"], action_idx))
                        nature_response = self.nature.action_simulator(env,\
                                                intervention_vars[action_idx])
                        episode_reward += nature_response["reward"]
                        logging.info(nature_response)
                        done = nature_response["done"]
                done = nature_response.get("done", False)
                self.rewards_per_round.append(episode_reward)
        return self.rewards_per_round

class LightsHalfBlindAgent(object):
    pass


if __name__ == "__main__":
    from utils.light_env_utils import generate_model_from_env
    from utils.vis_utils import plot_reward
    from true_causal_model import TrueCausalModelEnv
    env = LightEnv(structure="one_to_many")
    env.keep_struct = False
    env.reset()
    env.keep_struct = True
    model = generate_model_from_env(env)
    nature_light_switch = TrueCausalModelEnv(model)
    print(env.aj)
    print(env.goal)
    for cpd in model.pgmodel.get_cpds():
        print(cpd)
    episodes = 100
    print(model.conditional_probability("effect_1", {
          "cause_0": 1, "cause_1": 1, "cause_3": 0, "cause_4": 1}))
    print(model.conditional_probability("effect_1", {
          "cause_0": 1, "cause_1": 1, "cause_3": 0, "cause_4": 1}).values)
    agent = LightsFullyInformedAgent(nature=nature_light_switch, pgmodel=model)
    rewards = agent.training(env, episodes)
    print(rewards)
    std_dev = np.zeros(len(rewards))
    plot_reward(rewards, std_dev, "FullyInformedAgent", "Lights")

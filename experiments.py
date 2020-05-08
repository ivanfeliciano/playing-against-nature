import logging
import argparse

import numpy as np

from model import BaseModel
from utils import vis_utils
from true_causal_model import TrueCausalModel
from model import BaseModel
from agents.causal_agents import FullyInformedAgent, HalfBlindAgent
from agents.q_learning import QLearning
from agents.random_agent import RandomAgent


def run_experiments(n_experiments=1, rounds=10, agent_class=None, nature=None,\
                    pgmodel=None, name="agent", target_value=None, plot=True, epsilon=0.3):
    reward_all_experiments = []
    for experiment in range(n_experiments):
        np.random.seed(experiment)
        if name == "Qlearning":
            agent = agent_class(nature, epsilon)
            agent.training(rounds)
        if name == "HalfBlindAgent" or name == "FullyInformedAgent":
            agent = agent_class(nature, pgmodel)
            agent.training(rounds, target_value)
        if name == "Random":
            agent = agent_class(nature)
            agent.training(rounds)
        acc_reward = np.cumsum(agent.get_rewards())
        acc_reward = acc_reward / rounds
        reward_all_experiments.append(acc_reward)
    mean_reward = np.mean(reward_all_experiments, axis=0)
    std_dev_reward = np.std(reward_all_experiments, axis=0)
    if plot:
        vis_utils.plot_reward(mean_reward, std_dev_reward, name, "{}experiments_{}rounds".format(n_experiments, rounds))
    return (mean_reward, std_dev_reward)

def main():
    parser = argparse.ArgumentParser(description='Run experiments causal agents.')
    parser.add_argument("--config-file", type=str,
                        default="configs/model_parameters.json", help="Path to the configuration files.")
    parser.add_argument("--experiments", type=int, default=10,
                        help="# of experiments.")
    parser.add_argument("--rounds", type=int, default=20,
                        help="# of rounds per experiment.")
    parser.add_argument("--target-value", type=int, default=1,
                        help="Desired value for target variable.")
    parser.add_argument("--log-file", type=str,
                        default="logs/experiments.log", help="Path to the log files.")
    args = parser.parse_args()
    logs_path = args.log_file
    model_path = args.config_file
    rounds = args.rounds
    target_value = args.target_value
    n_experiments = args.experiments
    if logs_path:
        logging.basicConfig(filename=logs_path, filemode='w', level=logging.INFO)
    model = BaseModel(model_path)
    nature = TrueCausalModel(model)
    names = ["Qlearning", "HalfBlindAgent", "FullyInformedAgent", "Random"]
    agents = [QLearning, HalfBlindAgent, FullyInformedAgent, RandomAgent]
    measures = [None, None, None, None]
    for i in range(4):
        measures[i] = run_experiments(n_experiments, rounds, agents[i],\
            nature, model, names[i], target_value)
    vis_utils.plot_rewards_comparison(
        measures, rounds, names, "{}experiments_{}rounds".format(n_experiments, rounds))
if __name__ == "__main__":
    main()

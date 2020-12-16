import pickle
import numpy as np

from env.light_env import LightEnv
from policy import Policy
from utils.light_env_utils import init_q_table

class QLearningAgent(object):
    """
    Clase base para un agente de Q learning.
    Aquí se configura el ambiente, los parámetros y
    el flujo del aprendizaje.
    """
    def __init__(self, environment, policy, episodes=100, alpha=0.8, gamma=0.95):
        self.env = environment
        self.policy = policy
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.Q = init_q_table(env)
    def select_action(self, state):
        return self.policy.select_action(self.env, state, self.Q)
    def train(self, dict_filename, s):
        self.avg_reward = []
        results_data = dict()
        rewards_per_episode = []
        for episode in range(self.episodes):
            total_episode_reward = 0
            state = tuple(self.env.reset()[:self.env.num])
            done = False
            while not done:
                action = self.select_action(state)
                new_state, reward, done, info = self.env.step(action)
                new_state = tuple(new_state[:self.env.num])
                self.Q[state][action] = self.Q[state][action] + self.alpha * \
                                        (reward + self.gamma * np.max(self.Q[new_state]) -\
                                        self.Q[state][action])
                state = new_state
                total_episode_reward += reward
            rewards_per_episode.append(total_episode_reward)
        results_data[f"rewards_{s}"] = rewards_per_episode
        print(results_data)
        with open(dict_filename + ".pickle", "wb") as handle:
            pickle.dump(results_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return rewards_per_episode

if __name__ == "__main__":
    num = 5
    simulations = 5
    episodes = 500
    structures = ["one_to_one", "one_to_many", "many_to_one"]
    for structure in structures:
        env = LightEnv(num=num, structure=structure)
        for s in range(simulations):
            env.keep_struct = False
            env.reset()
            env.keep_struct = True
            policy = Policy(linear=False)
            q_agent = QLearningAgent(env, policy, episodes=episodes)
            rewards = q_agent.train(
                f"results/light-switches-q-learning-exp-decay/{structure}/{num}/mats/light_env_struct_{structure}_{s}", s=s)

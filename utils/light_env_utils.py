import logging
import itertools

import numpy as np

from model import BaseModel
from env.light_env import LightEnv
from utils.helpers import powerset


def obs_to_tuple(obs, n):
    return tuple(map(int, obs[:n]))

def generate_model_from_env(env, lights_off=False):
    aj_mat = env.aj
    aj_list, parents = aj_matrix_to_aj_list(aj_mat)
    nodes = list(aj_list.keys())
    ebunch = aj_list_to_ebunch(aj_list)
    if lights_off:
        cpdtables = generate_cpdtables_from_aj_list(parents, invert=True)
    else:
        cpdtables = generate_cpdtables_from_aj_list(parents)
    targets = sorted(parents.keys())
    interventions = ["cause_{}".format(i) for i in range(env.num + 1)]
    target_vals = env.goal
    data = dict()
    data["digraph"] = ebunch
    data["cpdtables"] = cpdtables
    data["target"] = targets
    data["nature_variables"] = []
    data["interventions"] = interventions
    data["target_vals"] = target_vals
    data["nodes"] = nodes
    return BaseModel(data=data)

def aj_matrix_to_aj_list(aj_mat):
    n = len(aj_mat)
    aj_list = dict()
    parents = {"effect_{}".format(i) : [] for i in range(n)}
    for i in range(n):
        cause_name = "cause_{}".format(i)
        effect_name = "effect_{}".format(i)
        aj_list[cause_name] = []
        aj_list[effect_name] = []
        for j in range(n):
            if aj_mat[i][j]:
                aj_list[cause_name].append("effect_{}".format(j))
                parents["effect_{}".format(j)].append(cause_name)
    aj_list["cause_{}".format(n)] = []
    return aj_list, parents
def aj_list_to_ebunch(aj_list):
    ebunch = []
    for node in aj_list:
        for child in aj_list[node]:
            ebunch.append((node, child))
    return ebunch
def generate_cpdtables_from_aj_list(parents, invert=False):
    cpdtables = []
    print(parents)
    for node in parents:
        variable_card = 2
        evidence = sorted(parents[node])
        values = []
        table = dict()
        if not evidence:
            values.append([0.5, 0.5])
        else:
            lights_on = []
            lights_off = []
            evidence_vals = [(0, 1) for _ in evidence]
            evidence_card = [2 for _ in evidence]
            for prod in itertools.product(*evidence_vals):
                if sum(prod) == 0:
                    lights_on.append(0)
                    lights_off.append(1)
                else:
                    lights_on.append(int(sum(prod) % 2))
                    lights_off.append(int(not sum(prod) % 2))
            if not invert:
                values = [lights_off, lights_on]
            else:
                values = [lights_on, lights_off]
            table["evidence"] = evidence
            table["evidence_card"] = evidence_card
        table["variable"] = node
        table["variable_card"] = variable_card
        table["values"] = values
        cpdtables.append(table)
    for cause in ["cause_{}".format(i) for i in range(len(parents) + 1)]:
        table = dict()
        table["variable"] = cause
        table["variable_card"] = 2
        table["values"] = [[0.5, 0.5]]
        cpdtables.append(table)
    return cpdtables


def check_diff_and_get_target_variables(env):
    num = env.num
    state = env._get_obs()[:num]
    goal = env.goal
    target_vars = []
    targets_vals = []
    for i in range(len(state)):
        if int(state[i]) != int(goal[i]):
            target_vars.append("effect_{}".format(i))
            targets_vals.append(goal[i])
    return target_vars, targets_vals

def generate_invalid_edges_light(variables, causes):
    invalid_edges = []
    for pair in itertools.combinations(causes, 2):
        invalid_edges.append(pair)
    for pair in itertools.combinations(variables, 2):
        if pair not in invalid_edges and pair[0][0] == "e" or pair[1][0] == "c":
            invalid_edges.append(pair)
    return invalid_edges


def explore_light_env(env, n_steps):
    data_dict_on = dict()
    data_dict_off = dict()
    for i in range(n_steps):
        action = env.action_space.sample()
        obs = action_simulator(env, "cause_{}".format(action))
        done = obs.pop("done", None)
        obs.pop("reward", None)
        change = obs.pop("change_to", None)
        for k in obs:
            if k not in data_dict_on:
                data_dict_on[k] = []
            if k not in data_dict_off:
                data_dict_off[k] = []
            if change == "on":
                data_dict_on[k].append(obs[k])
            elif change == "off":
                data_dict_off[k].append(obs[k])
            else:
                data_dict_on[k].append(obs[k])
                data_dict_off[k].append(obs[k])
        if done:
            env.reset()
    return data_dict_on, data_dict_off


def get_targets(env):
    """
    Retorna las variables cuyo valor es diferente al de la meta.
    """
    goal = env.goal
    obs = env._get_obs()[:env.num]
    targets = dict()
    for i in range(len(obs)):
        if int(obs[i]) != int(goal[i]):
            targets[f"effect_{i}"] = int(goal[i])
    return targets

def get_good_action(env, parents):
    targets, vals = check_diff_and_get_target_variables(env)
    if len(targets) > 0:
        target = np.random.choice(targets)
        if target in parents:
            return np.random.choice(parents[target])
    return f"cause_{env.num}"



def action_simulator(env, chosen_action):
    response = dict()
    action = int(chosen_action.strip().split("_")[-1])
    obs_before = env._get_obs()[:env.num]
    if action == env.num:
        response["cause_{}".format(env.num)] = 1
    else:
        response["cause_{}".format(env.num)] = 0
    obs, reward, done, info = env.step(action)
    response["reward"] = reward
    response["done"] = done
    for i in range(len(env.state)):
        response["cause_{}".format(i)] = int(env.state[i])
    for i in range(env.num):
        response["effect_{}".format(i)] = int(obs[i])
    response["change_to"] = "nothing"
    for i in range(len(obs_before)):
        if int(obs_before[i]) - int(obs[i]) < 0:
            response["change_to"] = "on"
        if int(obs_before[i]) - int(obs[i]) > 0:
            response["change_to"] = "off"
        
    return response


def init_q_table(env):
    all_states = powerset(env.num)
    return {
        state:  np.zeros(env.num + 1) for state in all_states
    }

def parents_from_ebunch(ebunch):
    """
    Obtiene un diccionario donde las llaves son nodos
    y los valores son listas con los padres de cada nodo.
    """
    parents = dict()
    for edge in ebunch:
        if not edge[1] in parents:
            parents[edge[1]] = []
        parents[edge[1]].append(edge[0])
    return parents

if __name__ == "__main__":
    n = 5
    logging.basicConfig(filename='logs/envToModel.log',
                        filemode='w', level=logging.INFO)
    env = LightEnv(structure="many_to_one")
    env.keep_struct = False
    print(env._get_obs()[:n])
    print(env.goal)
    env.reset()
    env.keep_struct = True
    print(env._get_obs()[:n])
    print(env.goal)
    env.reset()
    print(env._get_obs()[:n])
    print(env.goal)
    print(get_targets(env))
    lights_on_model = generate_model_from_env(env)
    lights_off_model = generate_model_from_env(env, lights_off=True)
    data_on, data_off = explore_light_env(env, 10)
    # print(data_on)
    # print(data_off)

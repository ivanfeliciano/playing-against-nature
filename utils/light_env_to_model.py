import logging
import itertools

from model import BaseModel
from env.light_env import LightEnv

def generate_model_from_env(env):
    aj_mat = env.aj
    aj_list, parents = aj_matrix_to_aj_list(aj_mat)
    nodes = list(aj_list.keys())
    ebunch = aj_list_to_ebunch(aj_list)
    cpdtables = generate_cpdtables_from_aj_list(parents)
    targets = sorted(parents.keys())
    interventions = ["cause_{}".format(i) for i in range(env.num)]
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
    return aj_list, parents
def aj_list_to_ebunch(aj_list):
    ebunch = []
    for node in aj_list:
        for child in aj_list[node]:
            ebunch.append((node, child))
    return ebunch
def generate_cpdtables_from_aj_list(parents):
    cpdtables = []
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
                lights_on.append(int(sum(prod) % 2))
                lights_off.append(int(not sum(prod) % 2))
            values = [lights_off, lights_on]
            table["evidence"] = evidence
            table["evidence_card"] = evidence_card
        table["variable"] = node
        table["variable_card"] = variable_card
        table["values"] = values
        cpdtables.append(table)
    for cause in ["cause_{}".format(i) for i in range(len(parents))]:
        table = dict()
        table["variable"] = cause
        table["variable_card"] = 2
        table["values"] = [[0.5, 0.5]]
        cpdtables.append(table)
    return cpdtables


if __name__ == "__main__":
    n = 5
    logging.basicConfig(filename='logs/envToModel.log',
                        filemode='w', level=logging.INFO)
    env = LightEnv(structure="one_to_many")
    env.keep_struct = False
    env.reset()
    env.keep_struct = True
    model = generate_model_from_env(env)
    print(env.step(0))

import random
import numpy as np
from copy import deepcopy 
def powerset(n):
    powerset = []
    for i in range(1 << n):
        powerset.append(tuple([int(_) for _ in np.binary_repr(i, width=n)]))
    return powerset

def aj_to_adj_list(adj_mat):
    n_causes = len(adj_mat)
    adj_list = dict()
    for (node, succesors) in enumerate(adj_mat):
        adj_list[node] = [(i + n_causes, succesors[i]) for i in range(n_causes) if succesors[i] > 0]
    return adj_list

def remove_edges(adj_list, prob=0.85):
    for node in adj_list:
        r = np.random.uniform()
        if r > prob:
            np.random.shuffle(adj_list[node])
            if len(adj_list[node]) > 0:
                adj_list[node].pop()
    return adj_list

def to_wrong_graph(adj_list, n_effects, prob=0.6):
    for node in adj_list:
        r = np.random.uniform()
        if prob / 2 < r < prob:
            np.random.shuffle(adj_list[node])
            if len(adj_list[node]) > 0:
                adj_list[node].pop()
        elif r < prob / 2:
            effects = np.arange(n_effects, 2 * n_effects)
            np.random.shuffle(effects)
            for effect in effects:
                in_effects = False
                for suc in adj_list[node]:
                    if effect == suc[0]: in_effects = True
                if not in_effects: 
                    adj_list[node].append((effect, 1))
                    break
    return adj_list
if __name__ == "__main__":
    a  = np.array([np.zeros(5) for _ in range(5)])
    for i in range(len(a)):
        for j in range(len(a[i])):
            if np.random.uniform() > 0.75:
                a[i, j] = 1
    print(a)
    adj_list = aj_to_adj_list(a)
    print("ADJ")
    print(adj_list)
    print("INCOMPLETA")
    adj_list_incompleta = remove_edges(deepcopy(adj_list))
    print(adj_list_incompleta)
    print("INCORRECTA")
    adj_list_incorrecta = to_wrong_graph(deepcopy(adj_list), len(a))
    print(adj_list_incorrecta)
    print("ADJ")
    print(adj_list)
import random

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class CausalStructure(object):
    def __init__(self, adj_list):
        self.graph = nx.DiGraph()
        self.causes = []
        for node in adj_list:
            for succesor in adj_list[node]:
                self.graph.add_edge(node, succesor[0], weight=succesor[1])
            self.causes.append(node)
    def get_structure(self):
        pass
    def draw_graph(self, filename="graph"):
        pos = nx.bipartite_layout(self.graph, self.causes)
        nx.draw(self.graph, pos=pos, with_labels=True)
        plt.savefig("{}.png".format(filename))
        # plt.show()
        plt.close()
    def get_causes(self, node, shuffle=True, threshold=0.5):
        if node not in self.graph:
            return []
        preds = list(self.graph.predecessors(node))
        preds = [i for i in preds if self.graph[i][node]["weight"] > threshold]
        if shuffle: random.shuffle(preds)
        return preds

if __name__ == "__main__":
    adj = {
        1: [(2, 1), (3, 1)],
        2: [(3, 1), (4, 1)],
        4: [(3, 1)],
        5: [(2, 1)]
    }
    graph = CausalStructure(adj)
    print(graph.get_causes(2))
    print(graph.get_causes(25))
    graph.draw_graph()
    
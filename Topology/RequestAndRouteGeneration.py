# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 28-07-2023                                      #
#      Goals: request and candidate route generation                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import HyperParameters
import numpy as np
import pandas as pd
import networkx as nx
import random

class RequestAndRouteGeneration:
    def __init__(self):
        self.nodes_num = HyperParameters.topology_myself_nodes_num
        self.requests = []
        self.volumn_upper = 10
        self.volumn_lower = 2
        self.requests = []
        self.candidate_routes = {}

    def request_generation(self):
        source = random.randint(1,self.nodes_num)
        destination = random.randint(1,self.nodes_num)
        volumn = random.randint(self.volumn_lower, self.volumn_upper)
        self.requests.append([source, destination, volumn])

    def route_generation(self):
        nodes_num = HyperParameters.topology_myself_nodes_num
        nodes = [i + 1 for i in range(nodes_num)]
        data = pd.read_csv(HyperParameters.topology_myself_data_path)
        G = nx.Graph()
        for node in nodes:
            G.add_node(node)
        node1 = data["node1"].values.tolist()
        node2 = data["node2"].values.tolist()
        length = data["length"].values.tolist()
        for i in range(len(node1)):
            G.add_edge(node1[i], node2[i], length=length[i])
        for index, r in self.requests:
            self.candidate_routes[index] = []
            paths = nx.shortest_simple_paths(G, r[0], r[1])
            for c,p in enumerate(paths):
                if c == 6:
                    break
                self.candidate_routes[index].append(p)

if __name__ == '__main__':
    r_r_g = RequestAndRouteGeneration()
    r_r_g.route_generation()





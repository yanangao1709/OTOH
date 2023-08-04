# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 28-07-2023                                      #
#      Goals: request and candidate route generation          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from Topology import HyperParameters as thp
import numpy as np
import pandas as pd
import networkx as nx
import random
import Request


class RequestAndRouteGeneration:
    def __init__(self):
        self.nodes_num = thp.topology_myself_nodes_num
        self.volumn_upper = thp.volumn_upper
        self.volumn_lower = thp.volumn_lower
        self.candidate_routes = {}

    def request_routes_generation(self):
        requests = []
        # candidate route generation
        candidate_routes = [[] for r in requests]
        nodes_num = thp.topology_myself_nodes_num
        nodes = [i + 1 for i in range(nodes_num)]
        data = pd.read_csv(thp.topology_myself_data_path)
        G = nx.Graph()
        for node in nodes:
            G.add_node(node)
        node1 = data["node1"].values.tolist()
        node2 = data["node2"].values.tolist()
        length = data["length"].values.tolist()
        for i in range(len(node1)):
            G.add_edge(node1[i], node2[i], length=length[i])
        for index, r in enumerate(requests):
            candidate_routes[index] = []
            paths = nx.shortest_simple_paths(G, r[0], r[1])
            for c, p in enumerate(paths):
                if c == 6:
                    break
                candidate_routes[index].append(p)

        # request generation
        for i in range(thp.request_num):
            r = Request()
            r.setSource(random.randint(1,self.nodes_num))
            r.setDestination(random.randint(1,self.nodes_num))
            r.setVolumn(self.volumn_lower, self.volumn_upper)
            r.setCandidateRoutes(candidate_routes[i])
            requests.append(r)
        return requests


# just for test
# if __name__ == '__main__':
#     r_r_g = RequestAndRouteGeneration()
#     requests = r_r_g.request_generation()
#     candidate_routes = r_r_g.route_generation(requests)





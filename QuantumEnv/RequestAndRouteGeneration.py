# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 28-07-2023                                      #
#      Goals: request and candidate route generation          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from TOQN import TOQNHyperparameters as tohp
from QuantumEnv.Request import Request
import pandas as pd
import networkx as nx
import random


class RequestAndRouteGeneration:
    def __init__(self):
        self.nodes_num = tohp.request_num
        self.volumn_upper = tohp.volumn_upper
        self.volumn_lower = tohp.volumn_lower
        self.candidate_routes = {}

    def request_routes_generation(self):
        requests = []
        # candidate route generation
        nodes_num = tohp.nodes_num
        nodes = [i + 1 for i in range(nodes_num)]
        data = pd.read_csv(tohp.topology_data_path)
        G = nx.Graph()
        for node in nodes:
            G.add_node(node)
        node1 = data["node1"].values.tolist()
        node2 = data["node2"].values.tolist()
        length = data["length"].values.tolist()
        for i in range(len(node1)):
            G.add_edge(node1[i], node2[i], length=length[i])

        # request generation
        for i in range(tohp.request_num):
            r = Request()
            candidate_routes = []
            can_route_hops = []
            # source is not same as destination
            nodes = [i+1 for i in range(self.nodes_num)]
            r.setSource(random.randint(1, self.nodes_num))
            nodes.remove(r.getSource())
            r.setDestination(nodes[random.randint(0,len(nodes)-1)])
            paths = nx.shortest_simple_paths(G, r.getSource(), r.getDestination())
            for c, p in enumerate(paths):
                if c == tohp.candidate_route_num:
                    break
                candidate_routes.append(p)
                can_route_hops.append(len(p))
            r.setVolumn(self.volumn_lower, self.volumn_upper)
            r.setCandidateRoutes(candidate_routes)
            r.setCandRouteHops(can_route_hops)
            requests.append(r)
        return requests


# just for test
# if __name__ == '__main__':
#     r_r_g = RequestAndRouteGeneration()
#     requests = r_r_g.request_generation()
#     candidate_routes = r_r_g.route_generation(requests)





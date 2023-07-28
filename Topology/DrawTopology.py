# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 28-07-2023                                      #
#      Goals: draw topology according to the data             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import networkx as nx
import matplotlib.pyplot as plt
import HyperParameters
import pandas as pd

def draw():
    nodes_num = HyperParameters.topology_myself_nodes_num
    nodes = [i+1 for i in range(nodes_num)]
    data = pd.read_csv(HyperParameters.topology_myself_data_path)
    G = nx.Graph()
    for node in nodes:
        G.add_node(node)
    node1 = data["node1"].values.tolist()
    node2 = data["node2"].values.tolist()
    length = data["length"].values.tolist()
    for i in range(len(node1)):
        r = G.add_edge(node1[i], node2[i], length = length[i])

    # paths = nx.shortest_simple_paths(G, 3,5)
    # for c,p in enumerate(paths):
    #     print(p)
    # print(nx.shortest_path_length(G, source=3, target=5))

    nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color='y',)
    plt.show()

if __name__ == '__main__':
    draw()

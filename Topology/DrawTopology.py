# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 28-07-2023                                      #
#      Goals: draw topology according to the data             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from Topology import TopoHyperParameters as tohp
import pandas as pd
matplotlib.use("TkAgg")

class DrawTopology:
    def __init__(self):
        self.node_num = tohp.topology_myself_nodes_num

    def getLinkLength(self):
        link_length = [[100] * self.node_num for i in range(self.node_num)]
        data = pd.read_csv(tohp.topology_myself_data_path, index_col=False)
        for d in data.values:
            link_length[d[0]-1][d[1]-1] = d[2]
        return link_length

    def getLinkLength(self):
        data = pd.read_csv(tohp.topology_myself_data_path)
        node1 = data["node1"].values.tolist()
        node2 = data["node2"].values.tolist()
        length = data["length"].values.tolist()
        link_lens = [[0] * self.node_num for i in range(self.node_num)]
        for i in range(len(node1)):
            link_lens[node1[i]-1][node2[i]-1] = length[i]
            link_lens[node2[i] - 1][node1[i] - 1] = length[i]
        print(link_lens)
        return link_lens

    def draw(self):
        nodes_num = tohp.topology_myself_nodes_num
        nodes = [i+1 for i in range(nodes_num)]
        data = pd.read_csv(tohp.topology_myself_data_path)
        G = nx.Graph()
        for node in nodes:
            G.add_node(node)
        node1 = data["node1"].values.tolist()
        node2 = data["node2"].values.tolist()
        length = data["length"].values.tolist()
        for i in range(len(node1)):
            G.add_edge(node1[i], node2[i], length = length[i])

        nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color='y',)
        plt.show()

# just for test
if __name__ == '__main__':
    testDraw = DrawTopology()
    testDraw.draw()
    # link_lens = testDraw.getLinkLength()

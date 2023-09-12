# import matplotlib.pyplot as plt
# import numpy as np
# import random
#
# def test():
#     x = [23, 71, 34, 84, 19, 49, 85, 25, 51, 80, 50, 40]
#     y = [26, 34, 23, 27, 13, 13, 7, 55, 71, 60, 50, 40]
#     c = [26, 34, 23, 27, 13, 13, 7, 55, 71, 60, 50, 40]
#     plt.scatter(x, y, s=30, c='b')
#     for i in range(0, len(x)):
#         plt.annotate('('+f'{x[i]}'+','+f'{y[i]}'+')', xy=(x[i], y[i]), xytext=(x[i] - 0.1, y[i] - 0.1))
#         # plt.annotate(y[i], xy=(x[i], y[i]), xytext=(x[i] + 3, y[i] - 0.1))
#     plt.show()

# Nodes = [1,2,3,4,5,6,7,8,9]
# Nodes_Capacity = [0,0,0,0,8,4,4,8,2]
# Link_Capacity = [[0,0,0,0,2,0,0,0,0],
#                  [0,0,0,0,0,0,2,0,0],
#                  [0,0,0,0,0,0,0,3,0],
#                  [0,0,0,0,0,0,0,0,1],
#                  [2,0,0,0,0,2,0,4,0],
#                  [0,0,0,0,2,0,2,2,1],
#                  [0,2,0,0,0,2,0,0,0],
#                  [0,0,3,0,4,2,0,0,0],
#                  [0,0,0,1,0,1,1,0,0]]
#
# Delay = [[0,0,0,0,2,0,0,0,0],
#          [0,0,0,0,0,0,1,0,0],
#          [0,0,0,0,0,0,0,1,0],
#          [0,0,0,0,0,0,0,0,1],
#          [2,0,0,0,0,3,1,1,0],
#          [0,0,0,0,3,0,3,2,2],
#          [0,1,0,0,0,3,0,0,2],
#          [0,0,1,0,1,2,0,0,0],
#          [0,0,0,1,0,2,2,0,0]]
#
# # <source, destination, volume>
# Requests_Set = [[1,2,16],[1,3,8],[1,4,9],[3,2,18],[3,4,6]]
# Paths_Set = [[[1,5,6,7,2],[1,5,4,6,7,2],[1,5,6,9,7,2],[1,5,4,6,9,7,2]],
#              [[1,5,8,3],[1,5,6,8,3]],
#              [[1,5,6,9,4],[1,5,6,7,9,4],[1,5,8,6,9,4],[1,5,8,6,7,9,4]],
#              [[3,8,6,7,2],[3,8,5,6,7,2],[3,8,6,9,7,2],[3,8,5,6,9,7,2]],
#              [[3,8,6,9,4],[3,8,5,6,9,4],[3,8,6,7,9,4],[3,8,5,6,7,9,4]]]

# dilicley distribution
# import numpy as np
# print(np.random.dirichlet(np.ones(5), size=1))

import math
v1 = (math.sqrt(0.1381*pow(math.e, -0.2))+ math.sqrt(0.8523*pow(math.e, -0.2)) \
    + math.sqrt(0.0096*pow(math.e, -0.2))) * 0.09*(4/8+9/15+8/17)
v2 = (math.sqrt(0.4842*pow(math.e, -0.2))+ math.sqrt(0.0192*pow(math.e, -0.2)) \
    + math.sqrt(0.183*pow(math.e, -0.2)) + math.sqrt(0.3126*pow(math.e, -0.2))
    + math.sqrt(0.01*pow(math.e, -0.2)) ) * 0.09*(4/7+4/21)
print(v2)


# class Nodes:
#     def __init__(self, node_num, capacity):
#         self.num = node_num
#         self.capacity = capacity
#
#
# link_capacity = [
#     [0,1,1,1,0],
#     [1,0,1,0,1],
#     [1,1,0,1,1],
#     [1,0,1,0,1],
#     [0,1,1,1,0]
# ]
#
# link_length= [
#     [0,6,8,5,5],
#     [6,0,10,0,7],
#     [8,10,0,9,0],
#     [5,0,9,0,3],
#     [5,7,0,3,0]
# ]
#
# requests = {
#     (2,4): [[2, 1, 4],[2, 5, 4],[2, 3, 4],[2, 1, 5, 4],[2, 1, 3, 4]],
#     (5,3): [[5, 1, 3],[5, 4, 3],[5, 2, 3],[5, 1, 4, 3],[5, 1, 2, 3]],
#     (2,5): [[2, 5],[2, 1, 5], [2, 3, 1, 5]],
#     (4,1): [[4, 1],[4, 3, 1],[4, 5, 1],[4, 3, 2, 1],[4, 5, 2, 1]],
#     (3,1): [[3, 1],[3, 2, 1],[3, 4, 1],[3, 2, 5, 1]]
# }
#
# import networkx as nx
# import matplotlib.pyplot as plt
# Nodes = [1, 2, 3, 4, 5]
# pos = {1:(5,5), 2:(3,9), 3:(12,6), 4:(4,1), 5:(2,3)}
# G = nx.Graph()
# for node in Nodes:
#     G.add_node(node)
# edges = [
#     (1,2),(1,3),(1,4),(1,5),
#     (2,5),(4,5),(3,4),(2,3)
# ]
# r=G.add_edges_from(edges)
#
# paths = nx.shortest_simple_paths(G, 3,5)
# for c,p in enumerate(paths):
#     print(p)
# print(nx.shortest_path_length(G, source=3, target=5))
#
# nx.draw(G, pos=pos, with_labels=True, node_color='y',)
# plt.show()


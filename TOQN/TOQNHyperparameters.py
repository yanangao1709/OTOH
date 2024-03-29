# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 22-08-2023                                      #
#      Goals: the hyperparameters for TOQN                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import os
request_num = 5 # 5 8 6 4
candidate_route_num = 3
nodes_num = 18
topology_data_path = os.getcwd() + "\\Topology\\data\\topology-myself\\topology.csv"
volumn_upper = 10
volumn_lower = 2

# fidelity threshold
F_thr = 0.00000001
# delay threshold
D_thr = 150

topology_myself_nodes_num = 18
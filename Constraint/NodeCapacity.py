# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 11-09-2023                                      #
#      Goals: node capacity calculation on t^th time          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from Constraint.PolicyStorage import StoragePolicy
from TOQN import TOQNHyperparameters as tohp
from TOQN import TOQNHyperparameters as tohp
from Topology.TOQNTopology import H_RKN

class NodeCap:
    def __init__(self):
        self.req_num = tohp.request_num
        test = 1

    def obtain_his_node_capacity(self, m, ps):
        occupied_photons = 0
        Y_his = ps.get_Y_his()
        M_his = ps.get_M_his()
        for t in range(len(Y_his)):
            for r in range(tohp.request_num):
                for k in range(tohp.candidate_route_num):
                    if Y_his[t][r][k] == 1 and H_RKN[r][k][m] == 1:
                        occupied_photons += M_his[t][r][m]
        return occupied_photons

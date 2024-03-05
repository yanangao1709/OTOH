# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: Calculate the fidelity, delay,                  #
#             capacity... constraints                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from Constraint.Fidelity import Fidelity
from Constraint.Delay import Delay
from Constraint.NodeCapacity import NodeCap
from ResourceAllocation import RLHyperparameters as RLhp

class Constraints:
    def __init__(self, ps):
        self.ps = ps
        self.episode_limit = RLhp.EPISODE_LIMIT

    def obatin_fidelity(self, r, k, M, t):
        f = Fidelity()
        return f.obtain_route_fidelity(r, k, M, t)

    def obtain_his_delay(self, r, t):
        d = Delay(self.episode_limit)
        return d.obtain_his_delay(r, t, self.ps)

    def obtain_node_cap(self, m):
        nc = NodeCap()
        return nc.obtain_his_node_capacity(m, self.ps)
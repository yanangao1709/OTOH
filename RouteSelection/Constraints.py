# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: Calculate the fidelity, delay,                  #
#             capacity... constraints                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from Common.Fidelity import Fidelity
from Common.Delay import Delay
from Common.NodeCapacity import NodeCap
class Constraints:
    def __init__(self, ps):
        self.ps = ps

    def obatin_fidelity(self, r, k, M, t):
        f = Fidelity()
        return f.obtain_route_fidelity(r, k, M, t)

    def obtain_delay(self, r, t):
        d = Delay()
        return d.obtain_delay(r, t, self.ps)

    def obtain_node_cap(self, m):
        nc = NodeCap()
        return nc.obtain_his_node_capacity(m, self.ps)
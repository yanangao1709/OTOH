# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: route delay calculation                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import numpy as np

from TOQN import TOQNHyperparameters as tohp
from Topology.TOQNTopology import LINK_LENS, ROUTE_LEN

class Delay:
    def __init__(self, episode_limit):
        test = 1

    def obtain_his_delay(self, r, t, ps):
        if t == 0:
            return 0
        Tr = ps.get_Tr()
        delay = 0
        for his_t in range(Tr[r]-1):
            test = ps.get_Y_his()[his_t, r].tolist()
            route = ps.get_Y_his()[his_t, r].tolist().index(1)
            delay += ROUTE_LEN[r][route]
        # print("step-----" + str(t) + "request----" + str(r) + "----delay----" + str(delay))
        return delay





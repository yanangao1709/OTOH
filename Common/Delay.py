# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: route delay calculation                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from Topology.TOQNTopology import D_VOLUMN
from TOQN import TOQNHyperparameters as tohp
from Common.PolicyStorage import StoragePolicy

class Delay:
    def __init__(self):
        self.req_memory = [0 for i in range(tohp.request_num)]

    def obtain_delay(self, r, t, ps):
        judge_throughput = ps.get_judge_throughput()
        r_throughput_his = ps.get_throughput_his()
        r_delay_his = 0
        for i in range(t):
            if judge_throughput[i] == 1:
                r_delay_his = sum(r_throughput_his[r][0:i])
                break
        return r_delay_his





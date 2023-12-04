# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 11-09-2023                                      #
#      Goals: storage route selection policy                  #
#             to calculate delay and throughput               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from TOQN import TOQNHyperparameters as tohp
from Topology.TOQNTopology import ROUTES, D_VOLUMN, LINK_LENS, HOPS
class StoragePolicy:
    def __init__(self):
        self.t_his = []
        self.Y_his = []
        self.M_his = []
        self.throughput_his = {}
        self.judge_throughput = [0 for i in range(tohp.request_num)]

    def storage_policy(self, Y, M, t):
        self.t_his.append(t)
        self.Y_his.append(Y)
        self.M_his.append(M)
        for r in range(tohp.request_num):
            if t == 0:
                self.throughput_his[r] = []
            if self.judge_throughput[r] == 1:
                self.throughput_his[r].append(0)
                continue
            selected_route = 0
            for k in range(tohp.candidate_route_num):
                if Y[r][k] == 0:
                    continue
                else:
                    selected_route = k
            route = ROUTES[r][selected_route]
            photon_allocated = 0
            for i in range(HOPS[r][selected_route]):
                if i==0:
                    continue
                photon_allocated += M[r][route[i]-1]
            self.throughput_his[r].append(Y[r][selected_route] * photon_allocated/HOPS[r][selected_route])
            if sum(self.throughput_his[r]) >= D_VOLUMN[r]:
                self.judge_throughput[r] = 1    # finish the transmission

    def get_last_Y_policy(self):
        return self.Y_his[len(self.Y_his)-1]

    def get_judge_throughput(self):
        return self.judge_throughput

    def get_throughput_his(self):
        return self.throughput_his

    def get_Y_his(self):
        return self.Y_his

    def get_M_his(self):
        return self.M_his


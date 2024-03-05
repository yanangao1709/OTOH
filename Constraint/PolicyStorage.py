# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 11-09-2023                                      #
#      Goals: storage route selection policy                  #
#             to calculate delay and throughput               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from TOQN import TOQNHyperparameters as tohp
from ResourceAllocation import RLHyperparameters as RLhp
from Topology.TOQNTopology import D_VOLUMN, HOPS, ROUTES
import numpy as np

class StoragePolicy:
    def __init__(self, episode_limit):
        self.episode_limit = episode_limit
        self.t = 0
        self.t_his = np.zeros(episode_limit, dtype=int)
        self.Y_his = np.zeros((episode_limit, tohp.request_num, RLhp.NUM_ACTIONS), dtype=int)
        self.M_his = np.zeros((episode_limit, tohp.request_num, tohp.nodes_num), dtype=int)
        self.throughput_his = {}
        self.judge_throughput = [False for i in range(tohp.request_num)]
        self.his_num = 0
        self.Tr = np.zeros(tohp.request_num, dtype=int)

    def storage_policy(self, Y, M, t):
        self.t = t
        self.t_his[t%self.episode_limit] = t
        self.Y_his[t%self.episode_limit, :, :] = Y
        self.M_his[t%self.episode_limit, :, :] = M
        for r in range(tohp.request_num):
            if self.his_num%self.episode_limit == 0:
                self.throughput_his[r] = np.zeros(self.episode_limit)
            selected_route = Y[r].index(1)
            route = ROUTES[r][selected_route]
            photon_allocated = 0
            for i in range(HOPS[r][selected_route]):
                photon_allocated += M[r][route[i]-1]
            self.throughput_his[r][self.his_num%self.episode_limit] = Y[r][selected_route] * photon_allocated/HOPS[r][selected_route]

            if sum(self.throughput_his[r][0:self.his_num%self.episode_limit]) >= D_VOLUMN[r]:
                self.judge_throughput[r] = True    # finish the transmission
            if not self.judge_throughput[r]:
                self.Tr[r] = self.his_num

        self.his_num += 1

    def get_last_Y_policy(self):
        return self.Y_his[self.t%self.episode_limit, :, :].tolist()

    def get_judge_throughput(self):
        return self.judge_throughput

    def get_throughput_his(self):
        return self.throughput_his

    def get_Y_his(self):
        return self.Y_his

    def get_M_his(self):
        return self.M_his

    def get_Tr(self):
        return self.Tr


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 04-08-2023                                      #
#      Goals: throughput calculation                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from TOQN import TOQNHyperparameters as tohp

class Thr:
    def __int__(self):
        self.throughput = 0
        self.his_thr = 0

    def calculate_thr(self, selectedRoute, photonAllocated, H_RKN):
        for r in range(tohp.request_num):
            for k in range(tohp.candidate_route_num):
                for m in range(tohp.nodes_num):
                    H_RKN[r][k][m] * photonAllocated[m][r]


        obj = quicksum(Y_vars[r][k] * quicksum(H_RKN[r][k][i] * self.M[r][i]
                                               for i in range(self.node_num)
                                               ) / HOPS[r][k]
                       for r in range(self.request_num)
                       for k in range(self.candidate_route_num)
                       )
        self.throughput +=
        self.his_thr += self.throughput

    def get_throuthput(self, selectedRoutes, photonAllocated, H_RKN):
        self.calculate_thr(selectedRoutes, photonAllocated)
        return self.throughput

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 04-08-2023                                      #
#      Goals: throughput calculation                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from TOQN import TOQNHyperparameters as tohp

class Thr:
    def __init__(self):
        self.his_thr = [0 for r in range(tohp.request_num)]

    def calculate_thr(self, requests, selectedRoute, photonAllocated, H_RKN):
        for r in range(tohp.request_num):
            totalPho = 0
            route = selectedRoute[r].index(1)
            for m in range(tohp.nodes_num):
                totalPho += H_RKN[r][route][m] * photonAllocated[m][r]
            r_thr = totalPho/requests[r].getCandRouteHops()[route]
            self.his_thr[r] += r_thr
        return sum(self.his_thr)

    def get_throuthput(self, requests, selectedRoutes, photonAllocated, H_RKN):
        total_thr = self.calculate_thr(requests, selectedRoutes, photonAllocated, H_RKN)
        return total_thr

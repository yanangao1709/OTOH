# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 04-08-2023                                      #
#      Goals: link fidelity calculation                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import math
from QuantumEnv import HyperParameters as qshp
from Topology.TOQNTopology import ROUTES, LINK_LENS, HOPS

GAMMA = 100
ETA = 10

class Fidelity:
    def __int__(self):
        test = 1

    def getLinkFidelity(self, i, j, X_i, X_j):
        multi_qe = MQE()
        alpha0_ij = multi_qe.redefine_assign_qstate_of_multiqubits(i, X_i, j, X_j)

    def obtain_route_fidelity(self, r, k, M, t):
        route = ROUTES[r][k]
        H = HOPS[r][k]
        mulM = 1
        sumM = 0
        sumLink = 0
        for i in range(H):
            if i==0:
                continue
            mulM *= M[r][route[i]-1]
            sumM += M[r][route[i]-1]
            sumLink += LINK_LENS[route[i-1]-1][route[i]-1]
        F = (pow(qshp.p, H) * pow(qshp.d, H/2) * mulM * pow((1-qshp.p), (qshp.d*sumM-H))
             * pow(math.e, -1*qshp.tau*sumLink*t))
        return F

if __name__ == '__main__':
    f = Fidelity()
    photonallocated = [
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    ]
    F = f.obtain_route_fidelity(1,2, photonallocated, 2)
    print(F)
    # f.getLinkFidelity(3, 5, 1,1) # length = 4


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: Main                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from RouteSelection.OptimalRoute import OptimalRS
from Common.PolicyStorage import StoragePolicy
from Common.Throughput import Thr
from ResourceAllocation.PhotonAllocation import PhotonAllocation
from TOQN import TOQNHyperparameters as tohp
from ResourceAllocation.DQNAgent import DQN
from ResourceAllocation.QNenv import QuantumNetwork as QN

T_thr = 100
EPISODES = T_thr

if __name__ == '__main__':
    opr = OptimalRS()
    ps = StoragePolicy()
    pa = PhotonAllocation()
    thr = Thr()

    # random photon allocation
    photonallocated = [
        [2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 6, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 3, 2, 2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 2, 8, 2]
    ]
    env = QN()
    agents = {}
    for m in range(tohp.nodes_num):
        local_env = QN()
        local_net = DQN()
        agents[m] = [local_env, local_net]
    for t in range(EPISODES):
        states = []
        for m in range(tohp.nodes_num):
            states.append(agents[m][0].reset())


        # route selection
        opr.set_photon_allocation(photonallocated)
        selected_route = opr.get_route_from_CRR(t, ps)
        ps.storage_policy(opr.get_Y(), photonallocated, t)

        # resource allocation
        pa.setSelectedRoute(selected_route)
        photonallocated.clear()
        photonallocated = pa.get_PApolicy()

        # calculate throughput
        t_thr = thr.get_throughput(selected_route, photonallocated)
        print("In the" + str(t) + " times transmission, the throughput is " + str(t_thr))






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
from QuantumEnv.QNEnv import QuantumNetwork as QN

T_thr = 100
EPISODES = T_thr
# random photon allocation
photonallocated = [
    [2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 6, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 3, 2, 2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 2, 8, 2]
]

if __name__ == '__main__':
    opr = OptimalRS()
    ps = StoragePolicy()
    pa = PhotonAllocation()

    env = QN()
    agents = {}
    for m in range(tohp.nodes_num):
        local_env = QN()
        local_net = DQN()
        agents[m] = [local_env, local_net]
    for t in range(EPISODES):
        states = env.reset()
        step_counter = 0
        total_reward = 0
        while True:
            step_counter += 1
            # route selection
            opr.set_photon_allocation(photonallocated)
            selected_route = opr.get_route_from_CRR(t, ps)
            ps.storage_policy(opr.get_Y(), photonallocated, t)
            env.setSelectedRoutes(selected_route)
            states = env.transformStates(states)

            actions = net.choose_action(states)

            if step_counter > T_thr:
                break




        # resource allocation
        pa.setSelectedRoute(selected_route)
        photonallocated.clear()
        photonallocated = pa.get_PApolicy()

        # calculate throughput
        t_thr = thr.get_throughput(selected_route, photonallocated)
        print("In the" + str(t) + " times transmission, the throughput is " + str(t_thr))






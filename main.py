# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: Main                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from RouteSelection.OptimalRoute import OptimalRS
from Common.PolicyStorage import StoragePolicy
from ResourceAllocation.PhotonAllocation import PhotonAllocation
from TOQN import TOQNHyperparameters as tohp
from ResourceAllocation import RLHyperparameters as RLhp
from ResourceAllocation.DQNAgent import DQN
from QuantumEnv.QNEnv import QuantumNetwork as QN
from ResourceAllocation.Agents import Agents

T_thr = 100
EPISODES = T_thr

if __name__ == '__main__':
    opr = OptimalRS()
    ps = StoragePolicy()
    pa = PhotonAllocation()

    env = QN()
    agents = Agents()
    for t in range(EPISODES):
        states, photonallocated = env.reset()
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
            # resource allocation
            actions = agents.choose_action(states)
            photonallocated = pa.get_PApolicy()
            next_states, reward, done = env.step(actions)
            agents.store_trans(states, actions, reward, next_states)
            total_reward += reward
            if net.memory_counter >= RLhp.MEMORY_CAPACITY:
                agents.learn()
            states = next_states

            if step_counter > T_thr:
                break
        print(episode)
        acc_reward.append(total_reward / step_counter)  # total_reward/step_counter
        axx.append(episode)
        net.plot(net.ax, acc_reward)
        print("In the" + str(t) + " times transmission, the throughput is " + str(t_thr))






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: Main                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import matplotlib.pyplot as plt
from RouteSelection.OptimalRoute import OptimalRS
from Common.PolicyStorage import StoragePolicy
from TOQN import TOQNHyperparameters as tohp
from ResourceAllocation import RLHyperparameters as RLhp
from ResourceAllocation.DQNAgent import DQN
from QuantumEnv.QNEnv import QuantumNetwork as QN
from ResourceAllocation.Agents import Agents

T_thr = 100
EPISODES = 1000


if __name__ == '__main__':
    opr = OptimalRS()
    ps = StoragePolicy()

    env = QN()
    agents = Agents()
    acc_reward = []
    axx = []
    for episode in range(EPISODES):
        states, photonallocated = env.reset()
        step_counter = 0
        total_reward = 0
        while True:
            step_counter += 1
            # route selection
            opr.set_photon_allocation(photonallocated)
            selected_route = opr.get_route_from_CRR(episode, ps)
            ps.storage_policy(opr.get_Y(), photonallocated, episode)
            env.setSelectedRoutes(selected_route)
            states = env.transformStates(states)
            # resource allocation
            actions = agents.choose_action(states)
            next_states, reward = env.step(actions)
            agents.store_trans(states, actions, reward, next_states)
            total_reward += reward
            if agents.memory_counter >= RLhp.MEMORY_CAPACITY:
                agents.learn()
            states = next_states
            photonallocated = agents.get_PApolicy(actions)
            print("------Step_counter is " + str(step_counter))
            if step_counter > T_thr:
                break
        print(episode)
        acc_reward.append(total_reward / step_counter)  # total_reward/step_counter
        axx.append(episode)
        agents.plot(agents.ax, acc_reward)
        print("In the" + str(episode) + " times transmission, the total throughput of reqSet R is " + str(total_reward))
    plt.xlabel("episodes")
    plt.ylabel("throughput")
    plt.plot(axx, acc_reward, 'b-')
    plt.show()
    res = {"x": axx, "acc_reward": acc_reward}
    pd.DataFrame(res).to_csv('./Results/throughput.csv', index=False)






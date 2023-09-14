# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: RL for the photon allocation                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ResourceAllocation import RLHyperparameters as RLhp


class RLMain:
    def __init__(self, selectedroute):
        self.sr = selectedroute
        self.agents = []



    def run(self):
        # env =QN()
        # net = DQN()
        # print("The DQN is collecting experience...")
        # acc_reward = []
        # axx = []
        for episode in range(RLhp.EPISODES):
            state = env.reset()
            step_counter = 0
            total_reward = 0
            while True:
                step_counter += 1
                action = net.choose_action(state)
                next_state, reward, done = env.step(action, step_counter)
                net.store_trans(state, action, reward, next_state)  # 记录当前这组数据

                total_reward += reward
                if net.memory_counter >= MEMORY_CAPACITY:  # 攒够数据一起学
                    net.learn()
                if done:
                    # step_counter_list.append(step_counter)
                    # net.plot(net.ax, step_counter_list)
                    break
                state = next_state
            # print("episode {}, the reward is {}".format(episode, round(total_reward / step_counter, 3)))
            print(episode)
            # total_reward *= 14  # throughput scale
            # total_reward *= 10  # delay scale
            total_reward *= 16  # NoCRR throughput
            acc_reward.append(total_reward / step_counter)  # total_reward/step_counter
            axx.append(episode)
            net.plot(net.ax, acc_reward)



def normalize(fidelity):
    b = np.array(fidelity)
    max_v = np.max(b)
    min_v = np.min(b)
    normalized_data = (b - min_v) / (max_v - min_v)
    return normalized_data

def main():
    net = Dqn(NUM_STATES, NUM_ACTIONS)
    print("The DQN is collecting experience...")
    acc_reward = []
    axx = []
    for episode in range(EPISODES):
        state = env.reset()
        step_counter = 0
        total_reward = 0
        while True:
            step_counter +=1
            # env.render()
            action = net.choose_action(state)
            next_state, reward, done = env.step(action, step_counter)
            net.store_trans(state, action, reward, next_state)#记录当前这组数据

            total_reward += reward
            if net.memory_counter >= MEMORY_CAPACITY: # 攒够数据一起学
                net.learn()
            if done:
                # step_counter_list.append(step_counter)
                # net.plot(net.ax, step_counter_list)
                break
            state = next_state
        # print("episode {}, the reward is {}".format(episode, round(total_reward / step_counter, 3)))
        print(episode)
        # total_reward *= 14  # throughput scale
        # total_reward *= 10  # delay scale
        total_reward *= 16  # NoCRR throughput
        acc_reward.append(total_reward / step_counter)  # total_reward/step_counter
        axx.append(episode)
        net.plot(net.ax, acc_reward)

    # fideity scale
    # acc_reward = normalize(acc_reward)

    plt.xlabel("episodes")
    plt.ylabel("throughput")
    plt.plot(axx,acc_reward, 'b-')
    plt.show()
    res = {"x":axx, "acc_reward": acc_reward}
    pd.DataFrame(res).to_csv('./OTiM2R/NoCRR-Throughput.csv', index=False)

if __name__ == '__main__':
    main()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: all Agents                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from ResourceAllocation.DQNAgent import DQN
from TOQN import TOQNHyperparameters as tohp
import matplotlib.pyplot as plt

class Agents:
    def __init__(self):
        self.agents = {}
        self.initial_allAgents()
        self.memory_counter = 0
        self.fig, self.ax = plt.subplots()

    def initial_allAgents(self):
        for m in range(tohp.nodes_num):
            self.agents[m] = DQN()

    def choose_action(self, states):
        actions = {}
        for m in range(tohp.nodes_num):
            action = self.agents[m].choose_action(states[m])
            actions[m] = action
        return actions

    def store_trans(self, states, actions, reward, next_states):
        if self.memory_counter % 500 == 0:
            print("The experience pool collects {} time experience".format(self.memory_counter))
        for m in range(tohp.nodes_num):
            self.agents[m].store_trans(states[m], actions[m], reward, next_states[m], self.memory_counter)
        self.memory_counter += 1

    def learn(self):
        for m in range(tohp.nodes_num):
            self.agents[m].learn()

    def get_PApolicy(self, actions):
        photonAllocated = []
        for r in range(tohp.request_num):
            pa_m = []
            for m in range(tohp.nodes_num):
                pa_m.append(actions[m][r]+1)
            photonAllocated.append(pa_m)
        return photonAllocated

    def plot(self, ax, x):
        ax.cla()
        ax.set_xlabel("episode")
        ax.set_ylabel("accumulated reward")
        ax.plot(x, 'b-')
        plt.pause(0.00001)
        if ax == 500:
            plt.show()

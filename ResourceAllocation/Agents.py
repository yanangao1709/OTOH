# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: all Agents                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from ResourceAllocation.DQNAgent import DQN
from TOQN import TOQNHyperparameters as tohp

class Agents:
    def __init__(self):
        self.agents = {}
        self.initial_allAgents()

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
        for m in range(tohp.nodes_num):
            self.agents[m].store_trans(states[m], action[m], reward, next_state[m])

    def learn(self):
        for m in range(tohp.nodes_num):
            self.agents[m].learn()

    def get_PApolicy(self):





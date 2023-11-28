# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: all Agents                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import torch
import random
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from TOQN import TOQNHyperparameters as tohp
import matplotlib.pyplot as plt
import numpy as np
from ResourceAllocation import RLHyperparameters as RLhp
from ResourceAllocation.reward_decomposition.decomposer import RewardDecomposer
from ResourceAllocation.reward_decomposition import decompose as decompose

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.set_seed(1)

        self.input_layer = nn.Linear(RLhp.NUM_STATES, 32)
        self.input_layer.weight.data.normal_(0, 0.1)

        self.hidden_layer1 = nn.Linear(32,64)
        self.hidden_layer1.weight.data.normal_(0, 0.1)
        self.hidden_layer2 = nn.Linear(64,32)
        self.hidden_layer2.weight.data.normal_(0, 0.1)

        self.req_layers = {}
        for r in range(tohp.request_num):
            r_layer = nn.Linear(32, 32)
            r_layer.weight.data.normal_(0, 0.1)
            r_candroute_layer = nn.Linear(32, RLhp.NUM_ACTIONS)
            r_candroute_layer.weight.data.normal_(0, 0.1)
            self.req_layers[r] = [r_layer, r_candroute_layer]

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        v = []
        for r in range(tohp.request_num):
            x = F.relu(self.req_layers[r][0](x))
            v.append(F.relu(self.req_layers[r][1](x)))
        return tuple(v)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

class Agents:
    def __init__(self):
        self.memory_counter = 0
        self.fig, self.ax = plt.subplots()
        self.reward_decomposer = None

        self.eval_nets = {}
        self.target_nets = {}
        for i in range(tohp.nodes_num):
            self.eval_nets[i] = Net()
            self.target_nets[i] = Net()
        # storage data of state, action ,reward and next state
        self.memories = {}
        for i in range(tohp.nodes_num):
            self.memories[i] = np.zeros((RLhp.MEMORY_CAPACITY, 42))

        self.learn_counter = 0
        self.optimizers = {}
        self.losses = {}
        for i in range(tohp.nodes_num):
            self.optimizers[i] = Adam(self.eval_nets[i].parameters(), RLhp.LR)
            self.losses[i] = nn.MSELoss()

        self.reward_decomposer = RewardDecomposer()
        self.reward_optimiser = Adam(self.reward_decomposer.parameters(), lr=0.01)

    def choose_action(self, states):
        actions = {}
        for m in range(tohp.nodes_num):
            # notation that the function return the action's index nor the real action
            state = torch.unsqueeze(torch.FloatTensor(states[m]), 0)
            action = []
            if np.random.randn() <= RLhp.EPSILON:
                action_value = self.eval_nets[m].forward(state)
                for av in action_value:
                    a = torch.max(av, 1)[1].data.item()
                    action.append(a)
            else:
                for i in range(tohp.request_num):
                    a = np.random.randint(0, RLhp.NUM_ACTIONS)
                    action.append(a)
            actions[m] = action
        return actions

    def store_trans(self, states, actions, rewards, next_states):
        if self.memory_counter % 500 == 0:
            print("The experience pool collects {} time experience".format(self.memory_counter))
        for m in range(tohp.nodes_num):
            index = self.memory_counter % RLhp.MEMORY_CAPACITY
            trans = np.hstack((states[m], actions[m], rewards[m], next_states[m]))
            self.memories[m][index,] = trans
        self.memory_counter += 1

    def learn(self):
        # sample data
        sample_index = np.random.choice(RLhp.MEMORY_CAPACITY, RLhp.BATCH_SIZE)
        batch_memories = {}
        for m in range(tohp.nodes_num):
            batch_memories[m] = self.memories[m][sample_index, :]
        # train the agent
        self.train(batch_memories)
        # train the decomposer
        decompose.train_decomposer(self.reward_decomposer, batch_memories, self.reward_optimiser)

    def train(self, EpisodeBatch):
        # copy parameters to target each 100 episodes
        for m in range(tohp.nodes_num):
            if self.learn_counter % RLhp.Q_NETWORK_ITERATION == 0:
                self.target_nets[m].load_state_dict(self.eval_nets[m].state_dict())
        self.learn_counter += 1

        batch_rewards = self.build_rewards(EpisodeBatch)

        for i in range(tohp.nodes_num):
            batch_state = torch.FloatTensor(EpisodeBatch[i][:, :RLhp.NUM_STATES])
            batch_action = torch.LongTensor(EpisodeBatch[i][:, RLhp.NUM_STATES:RLhp.NUM_STATES + 1].astype(int))
            batch_next_state = torch.FloatTensor(EpisodeBatch[i][:, -RLhp.NUM_STATES:])
            q_eval_total = []
            for bs in self.eval_nets[i](batch_state):
                q_eval_total.append(bs.gather(1, batch_action))
            q_eval = sum(q_eval_total) / len(q_eval_total)

            q_next_total = []
            for bs in self.eval_nets[i](batch_next_state):
                q_next_total.append(bs.gather(1, batch_action))
            q_next = sum(q_next_total) / len(q_next_total)
            q_target = batch_rewards[:,:,i] + RLhp.GAMMA * q_next.max(1)[0].view(RLhp.BATCH_SIZE, 1)

            loss = self.losses[i](q_eval, q_target)
            self.optimizers[i].zero_grad()
            loss.backward(retain_graph=True)
            self.optimizers[i].step()

    def build_rewards(self, batch_memories):
        local_rewards = decompose.decompose(self.reward_decomposer, batch_memories)
        return local_rewards

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

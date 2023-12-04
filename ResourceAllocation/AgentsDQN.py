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

# device = torch.device("cpu" if torch.cuda.is_available() else "cuda")
device = torch.device("cpu")

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
            r_layer = nn.Linear(32, 32).to(device)
            r_layer.weight.data.normal_(0, 0.1)
            r_candroute_layer = nn.Linear(32, RLhp.NUM_ACTIONS).to(device)
            r_candroute_layer.weight.data.normal_(0, 0.1)
            self.req_layers[r] = [r_layer, r_candroute_layer]

    def forward(self, x):
        x = F.relu(self.input_layer(x)).to(device)
        x = F.relu(self.hidden_layer1(x)).to(device)
        x = F.relu(self.hidden_layer2(x)).to(device)
        v = []
        for r in range(tohp.request_num):
            x = F.relu(self.req_layers[r][0](x).to(device))
            v.append(F.relu(self.req_layers[r][1](x).to(device)))
        return tuple(v)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

class AgentsDQN:
    def __init__(self):
        self.memory_counter = 0
        self.fig, self.ax = plt.subplots()
        self.reward_decomposer = None

        self.eval_nets = {}
        self.target_nets = {}
        for i in range(tohp.nodes_num):
            self.eval_nets[i] = Net().to(device)
            self.target_nets[i] = Net().to(device)
        # storage data of state, action ,reward and next state
        self.memories = {}
        for i in range(tohp.nodes_num):
            self.memories[i] = np.zeros((RLhp.MEMORY_CAPACITY, 42))

        self.learn_counter = 0
        self.optimizers = {}
        self.losses = {}
        for i in range(tohp.nodes_num):
            self.optimizers[i] = Adam(self.eval_nets[i].parameters(), RLhp.LR)
            self.losses[i] = nn.MSELoss().to(device)

        self.reward_decomposer = RewardDecomposer()
        self.reward_optimiser = Adam(self.reward_decomposer.parameters(), lr=0.01)

    def choose_actionDQN(self, states):
        actions = {}
        for m in range(tohp.nodes_num):
            # notation that the function return the action's index nor the real action
            state = torch.unsqueeze(torch.FloatTensor(states[m]), 0).to(device)
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

        batch_rewards = self.build_rewards(EpisodeBatch).to(device)

        for i in range(tohp.nodes_num):
            batch_state = torch.FloatTensor(EpisodeBatch[i][:, :RLhp.NUM_STATES]).to(device)
            batch_action = torch.LongTensor(EpisodeBatch[i][:, RLhp.NUM_STATES:RLhp.NUM_STATES + tohp.request_num].astype(int)).to(device)
            batch_next_state = torch.FloatTensor(EpisodeBatch[i][:, -RLhp.NUM_STATES:]).to(device)
            # batch_rewards = torch.FloatTensor(EpisodeBatch[i][:,RLhp.NUM_STATES + tohp.request_num:RLhp.NUM_STATES + tohp.request_num+1])

            q_eval = torch.reshape(torch.stack(self.eval_nets[i](batch_state)), [RLhp.BATCH_SIZE, tohp.request_num, RLhp.NUM_ACTIONS]).gather(2, torch.reshape(batch_action, [RLhp.BATCH_SIZE,tohp.request_num,1]))
            q_next = torch.reshape(torch.stack(self.target_nets[i](batch_next_state)), [RLhp.BATCH_SIZE, tohp.request_num, RLhp.NUM_ACTIONS]).to(device)
            q_next_max = torch.reshape(torch.max(q_next, 2).values, [RLhp.BATCH_SIZE, tohp.request_num, 1])
            q_target = q_next_max.clone().to(device)
            for r in range(tohp.request_num):
                q_target[:,r,:] = batch_rewards[:,:,i] + RLhp.GAMMA * q_next_max[:,r,:]
                # q_target[:,r,:] = batch_rewards + RLhp.GAMMA * q_next_max[:,r, :]

            loss = self.losses[i](q_eval, q_target).to(device)
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

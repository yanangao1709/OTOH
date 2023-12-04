import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

import random
from torch.optim import Adam
from TOQN import TOQNHyperparameters as tohp
import matplotlib.pyplot as plt
from ResourceAllocation import RLHyperparameters as RLhp
from ResourceAllocation.reward_decomposition.decomposer import RewardDecomposer
from ResourceAllocation.reward_decomposition import decompose as decompose
from torch.distributions import Categorical

# device = torch.device("cpu" if torch.cuda.is_available() else "cuda")
device = torch.device("cpu")

from collections import deque, namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'policy'))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Actor(nn.Module):
    def __init__(self, action_dim, state_dim):
        super(Actor, self).__init__()
        set_seed(1)

        self.input_layer = nn.Linear(RLhp.NUM_STATES, 32)
        self.input_layer.weight.data.normal_(0, 0.1)

        self.hidden_layer1 = nn.Linear(32, 64)
        self.hidden_layer1.weight.data.normal_(0, 0.1)
        self.hidden_layer2 = nn.Linear(64, 32)
        self.hidden_layer2.weight.data.normal_(0, 0.1)

        self.req_layers = {}
        for r in range(tohp.request_num):
            r_layer = nn.Linear(32, 32)
            r_layer.weight.data.normal_(0, 0.1)
            r_candroute_layer = nn.Linear(32, RLhp.NUM_ACTIONS)
            r_candroute_layer.weight.data.normal_(0, 0.1)
            self.req_layers[r] = [r_layer, r_candroute_layer]

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x_share = F.relu(self.hidden_layer2(x))
        v = []
        for r in range(tohp.request_num):
            x_self = F.relu(self.req_layers[r][0](x_share))
            v.append(torch.sigmoid(self.req_layers[r][1](x_self)))
        return tuple(v)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        set_seed(1)

        self.input_layer = nn.Linear(RLhp.NUM_STATES, 32)
        self.input_layer.weight.data.normal_(0, 0.1)

        self.hidden_layer1 = nn.Linear(32, 64)
        self.hidden_layer1.weight.data.normal_(0, 0.1)
        self.hidden_layer2 = nn.Linear(64, 32)
        self.hidden_layer2.weight.data.normal_(0, 0.1)

        self.output_layer = nn.Linear(32, 1)
        self.output_layer.weight.data.normal_(0, 0.1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        x = F.relu(self.input_layer(x)).detach()
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        x = F.relu(self.output_layer(x))
        return x

class AgentsAC:
    def __init__(self):
        self.action_dim = RLhp.NUM_ACTIONS
        self.state_dim = RLhp.NUM_STATES
        self.memory_counter = 0
        self.learn_counter = 0
        self.fig, self.ax = plt.subplots()
        self.reward_decomposer = None

        self.actors = {}
        self.critics = {}
        self.actor_optims = {}
        self.critic_optims = {}
        self.losses = {}
        for i in range(tohp.nodes_num):
            self.actors[i] = Actor(self.action_dim, self.state_dim).to(device)
            self.critics[i] = Critic(self.state_dim).to(device)
            self.actor_optims[i] = torch.optim.Adam(self.actors[i].parameters(), lr=RLhp.lr_a)
            self.critic_optims[i] = torch.optim.Adam(self.critics[i].parameters(), lr=RLhp.lr_c)
            self.losses[i] = nn.MSELoss().to(device)

        # storage data of state, action ,reward and next state
        self.memories = {}
        for i in range(tohp.nodes_num):
            self.memories[i] = deque(maxlen=RLhp.MEMORY_CAPACITY)

        self.reward_decomposer = RewardDecomposer()
        self.reward_optimiser = Adam(self.reward_decomposer.parameters(), lr=0.01)

    def choose_actionAC(self, states, episode, step_counter):
        actions = {}
        log_probs = {}
        for m in range(tohp.nodes_num):
            state = torch.unsqueeze(torch.FloatTensor(states[m]), 0).to(device)
            actions[m] = []
            action_value = self.actors[m].forward(state)
            log_probs[m] = []
            for i in range(len(action_value)):
                av = action_value[i]
                dist = Categorical(av)
                a = dist.sample()
                temp = dist.log_prob(a)
                log_probs[m].append(dist.log_prob(a))
                actions[m].append(a.item())
        return actions, log_probs

    def store_trans(self, states, log_probs, rewards, next_states):
        if self.memory_counter % 500 == 0:
            print("The experience pool collects {} time experience".format(self.memory_counter))
        for m in range(tohp.nodes_num):
            index = self.memory_counter % RLhp.MEMORY_CAPACITY
            if len(self.memories[m]) == RLhp.MEMORY_CAPACITY:
                self.memories[m][index] = Transition(states[m], log_probs[m], rewards[m], next_states[m])
            else:
                self.memories[m].append(Transition(states[m], log_probs[m], rewards[m], next_states[m]))
            # self.log_prob_memory[m][index,] = log_probs[m]
        self.memory_counter += 1

    def learn(self):
        # sample data
        batch_memories = {}
        batch_log_probs = {}
        for m in range(tohp.nodes_num):
            sample_index = np.random.choice(RLhp.MEMORY_CAPACITY, RLhp.BATCH_SIZE)
            batch_memories[m] = np.zeros((RLhp.BATCH_SIZE,4), dtype=object)
            for i in range(len(sample_index)):
                batch_memories[m][i,:]= self.memories[m][sample_index[i]]
        # train the agent
        self.train(batch_memories)
        # train the decomposer
        decompose.train_decomposer(self.reward_decomposer, batch_memories, self.reward_optimiser)

    def train(self, EpisodeBatch):
        self.learn_counter += 1
        batch_rewards = self.build_rewards(EpisodeBatch).to(device)

        for i in range(tohp.nodes_num):
            batch_state = np.zeros((RLhp.BATCH_SIZE, RLhp.NUM_STATES))
            for j in range(RLhp.BATCH_SIZE):
                batch_state[j,:] = EpisodeBatch[i][:,0][j]
            batch_state_ts = torch.FloatTensor(batch_state).to(device)

            batch_next_state = np.zeros((RLhp.BATCH_SIZE, RLhp.NUM_STATES))
            for j in range(RLhp.BATCH_SIZE):
                batch_next_state[j, :] = EpisodeBatch[i][:, 3][j]
            batch_next_state_ts = torch.FloatTensor(batch_next_state).to(device)
            v = self.critics[i](batch_state_ts)
            v_ = self.critics[i](batch_next_state_ts)

            critic_loss = self.losses[i](RLhp.GAMMA * v_ + batch_rewards[:,:,i], v)
            self.critic_optims[i].zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optims[i].step()

            batch_log_prob = EpisodeBatch[i][:, 1]
            td = RLhp.GAMMA * v_ + batch_rewards[:,:,i] - v  # 计算TD误差
            self.actor_optims[i].zero_grad()

            tensor_list = []
            for b in range(RLhp.BATCH_SIZE):
                batch_log_prob_total = 0
                for j in range(tohp.request_num):
                    batch_log_prob_total += batch_log_prob[b][j]
                tensor_list.append(batch_log_prob_total.unsqueeze(1))
            batch_log_prob_list = torch.cat(tensor_list, dim=0)

            loss_actor = (-batch_log_prob_list.clone() * td.detach()).sum()
            with torch.autograd.set_detect_anomaly(True):
                loss_actor.backward(retain_graph=True)
            self.actor_optims[i].step()

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

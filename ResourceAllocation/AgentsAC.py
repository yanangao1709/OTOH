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


def set_seed(self, seed):
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
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        v = []
        for r in range(tohp.request_num):
            x = F.relu(self.req_layers[r][0](x))
            v.append(F.relu(self.req_layers[r][1](x)))
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

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        x = self.ln(F.relu(self.fc1(x)))
        return x


class AgentsACs:
    def __init__(self, env):
        self.gamma = 0.99
        self.lr_a = 3e-4
        self.lr_c = 5e-4

        self.env = env
        self.action_dim = self.env.action_space.n             #获取描述行动的数据维度
        self.state_dim = self.env.observation_space.shape[0]  #获取描述环境的数据维度

        self.actor = Actor(self.action_dim, self.state_dim)   #创建演员网络
        self.critic = Critic(self.state_dim)                  #创建评论家网络

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.loss = nn.MSELoss()

    def get_action(self, s):
        a = self.actor(s)
        dist = Categorical(a)
        action = dist.sample()             #可采取的action
        log_prob = dist.log_prob(action)   #每种action的概率

        return action.detach().numpy(), log_prob

    def learn(self, log_prob, s, s_, rew):
        #使用Critic网络估计状态值
        v = self.critic(s)
        v_ = self.critic(s_)

        critic_loss = self.loss(self.gamma * v_ + rew, v)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        td = self.gamma * v_ + rew - v          #计算TD误差
        loss_actor = -log_prob * td.detach()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

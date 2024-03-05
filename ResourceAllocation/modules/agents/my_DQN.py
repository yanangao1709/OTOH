import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ResourceAllocation import RLHyperparameters as RLhp
from TOQN import TOQNHyperparameters as tohp
import numpy as np
import torch
import random


class MyDQN(nn.Module):
    def __init__(self, input_shape, args):
        super(MyDQN, self).__init__()
        self.set_seed(1)

        self.input_layer = nn.Linear(RLhp.NUM_STATES, 32)
        self.input_layer.weight.data.normal_(0, 0.1)

        self.hidden_layer1 = nn.Linear(32,64)
        self.hidden_layer1.weight.data.normal_(0, 0.1)
        self.hidden_layer2 = nn.Linear(64,32)
        self.hidden_layer2.weight.data.normal_(0, 0.1)

        self.req_layers = {}
        self.device = torch.device('cuda:0')
        for r in range(tohp.request_num):
            r_layer = nn.Linear(32, 32).to(self.device)
            r_layer.weight.data.normal_(0, 0.1)
            r_candroute_layer = nn.Linear(32, RLhp.NUM_ACTIONS).to(self.device)
            r_candroute_layer.weight.data.normal_(0, 0.1)
            self.req_layers[r] = [r_layer, r_candroute_layer]

        self.ln = nn.LayerNorm(32)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x_share = F.relu(self.hidden_layer2(x))
        v = []
        for r in range(tohp.request_num):
            x_self = self.ln(F.relu(self.req_layers[r][0](x_share)))
            v.append(F.softmax(self.req_layers[r][1](x_self), dim=-1))
        return tuple(v)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

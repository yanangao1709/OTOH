# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: Each agent is first equipped with DQN RL model  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from torch import optim
from matplotlib import pyplot as plt
from TOQN import TOQNHyperparameters as tohp
from ResourceAllocation import RLHyperparameters as RLhp

class Net(nn.Module):
    def __init__(self):
        # 在Net类中调用其父类的__init__方法。
        # 这是为了继承父类的属性和方法，并初始化父类中定义的变量或对象。
        # super()函数可以避免直接引用父类名，从而更加灵活和通用。
        super(Net, self).__init__()
        self.set_seed(1)

        # self.fc1的输入维度为NUM_STATES，输出维度为30，
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

class DQN():
    def __init__(self):
        # 主网络eval_net（Q function/Q-table）
        # target网络target_net，记忆
        # eval_net 用于评估当前状态和动作之间的 Q 值，
        # 而 target_net 用于评估下一个状态和动作之间的 Q 值
        self.eval_net, self.target_net = Net(), Net()
        # 存数据
        # self.memory = np.zeros((MEMORY_CAPACITY, 17))
        self.memory = np.zeros((RLhp.MEMORY_CAPACITY, 84))
        # state, action ,reward and next state

        self.learn_counter = 0
        self.optimizer = optim.Adam(self.eval_net.parameters(), RLhp.LR) # 优化器：针对主网络进行更新
        self.loss = nn.MSELoss() # 回归

    def store_trans(self, state, action, reward, next_state, memory_counter):
        index = memory_counter % RLhp.MEMORY_CAPACITY
        trans = np.hstack((state, action, [reward], next_state))#记录一条数据
        self.memory[index,] = trans

    def choose_action(self, state_para):
        action = []
        # notation that the function return the action's index nor the real action
        state = torch.unsqueeze(torch.FloatTensor(state_para) ,0)
        if np.random.randn() <= RLhp.EPSILON:
            action_value = self.eval_net.forward(state)
            for av in action_value:
                a = torch.max(av, 1)[1].data.item()
                action.append(a)
        else: # 随机
            for i in range(tohp.request_num):
                a = np.random.randint(0,RLhp.NUM_ACTIONS)
                action.append(a)
        return action

    def learn(self):
        # 每学习100次之后，重新对target网络赋值
        if self.learn_counter % RLhp.Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())     #  学了100次之后target才更新（直接加载eval的权重）
        self.learn_counter+=1

        sample_index = np.random.choice(RLhp.MEMORY_CAPACITY, RLhp.BATCH_SIZE)    # 获取一个batch数据

        batch_memory = self.memory[sample_index, :]

        batch_state = torch.FloatTensor(batch_memory[:, :RLhp.NUM_STATES])
        # note that the action must be a int
        batch_action = torch.LongTensor(batch_memory[:, RLhp.NUM_STATES:RLhp.NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, RLhp.NUM_STATES+1: RLhp.NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -RLhp.NUM_STATES:])

        q_eval_total = []
        for bs in self.eval_net(batch_state):
            q_eval_total.append(bs.gather(1, batch_action))
        q_eval = sum(q_eval_total)/len(q_eval_total)
        # q_eval = self.eval_net(batch_state).gather(1, batch_action) # 得到当前Q(s,a)

        q_next_total = []
        for bs in self.eval_net(batch_next_state):
            q_next_total.append(bs.gather(1, batch_action))
        q_next = sum(q_next_total) / len(q_next_total)
        # q_next = self.target_net(batch_next_state).detach() # 得到Q(s',a')，有三个值，下面选max
        q_target = batch_reward + RLhp.GAMMA*q_next.max(1)[0].view(RLhp.BATCH_SIZE, 1) # bellman公式：Q=R+折扣*Q‘

        loss = self.loss(q_eval, q_target) # 差异越小越好
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() # 梯度更新

# ------test------
if __name__=="__main__":
    a = [3,5,6]
    print(a.values)
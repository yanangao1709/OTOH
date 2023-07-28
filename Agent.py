# ------actor-critic------repeater node agent-------
# ------------------Yanan Gao 24052023------------------
import random
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import HyperParameters as hp


class Net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Net, self).__init__()
        self.set_seed(1)

        self.input_layer = nn.Linear(input_shape, 32)
        self.hidden1 = nn.Linear(32, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.pi = nn.Linear(64, n_actions)
        self.V = nn.Linear(64, 1)
        self.optimizer = optim.Adam(self.parameters(), lr = hp.LR)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        pi = F.softmax(self.pi(x), dim=1)
        V = self.V(x)
        return pi, V

    def set_seed(self, seed):
        T.manual_seed(seed)
        T.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        T.backends.cudnn.deterministic = True


class A2CAgent():
    def __init__(self):
        self.agents = []
        self.memories = []
        self.log_prob = []
        for i in range(hp.AGENT_NUM):
            self.agents.append(Net(hp.N_OBSERVATIONS, hp.N_ACTIONS))
            self.memories.append(np.zeros((hp.MEMORY_CAPACITY, hp.RECORD_LENGTH)))
            self.log_prob.append(0)

        self.n_actions = hp.N_ACTIONS
        self.n_observation = hp.N_OBSERVATIONS
        self.memory_counter = 0
        self.step_counter = 0
        self.memory_capacity = hp.MEMORY_CAPACITY

        self.fig, self.ax = plt.subplots()

    def store_trans(self, obss, actions, reward, next_obss):
        if self.memory_counter % 500 == 0:
            print("The experience pool collects {} time experience".format(self.memory_counter))
        index = self.memory_counter % hp.MEMORY_CAPACITY
        for agent in range(hp.AGENT_NUM):
            trans = np.hstack((obss[agent], actions[agent], [reward], next_obss[agent]))
            self.memories[agent][index, ] = trans
        self.memory_counter += 1

    def choose_action(self, observations):
        actions = []

        for agent in range(hp.AGENT_NUM):
            observation = observations[agent]
            obs = T.tensor([observation], dtype=T.float)
            probabilities, _ = self.agents[agent](obs)
            action_probs = T.distributions.Categorical(probabilities)
            self.log_prob[agent] = action_probs.log_prob(action_probs.sample())

            if np.random.rand() <= hp.EPSILON:
                action = T.max(probabilities, 1)[1].data.item()
            else:
                for i in range(hp.REQUEST_NUM):
                    action = np.random.randint(0, hp.N_ACTIONS)
            actions.append(action)

        return actions

    def learn(self):
        for agent in range(hp.AGENT_NUM):
            self.agents[agent].optimizer.zero_grad()

            sample_index = np.random.choice(hp.MEMORY_CAPACITY, hp.BATCH_SIZE)  # 获取一个batch数据
            batch_memory = self.memories[agent][sample_index, :]

            batch_obs = T.FloatTensor(batch_memory[:, :self.n_observation])
            # note that the action must be a int
            batch_act = T.LongTensor(batch_memory[:, self.n_observation:self.n_observation + 1].astype(int))
            batch_reward = T.FloatTensor(batch_memory[:, self.n_observation + 1: self.n_observation + 2])
            batch_next_obs = T.FloatTensor(batch_memory[:, -self.n_observation:])

            _, critic_value = self.agents[agent].forward(batch_obs)
            _, critic_value_ = self.agents[agent].forward(batch_next_obs)

            delta = batch_reward + hp.GAMMA * critic_value_ - critic_value

            actor_total_loss = -self.log_prob[agent] * delta
            actor_loss = actor_total_loss
            critic_loss = delta ** 2

            loss = actor_loss + critic_loss
            T.mean(loss).backward()
            self.agents[agent].optimizer.step()

    def plot(self, ax, x):
        ax.cla()
        ax.set_xlabel("episode")
        ax.set_ylabel("accumulated reward")
        ax.plot(x, 'b-')
        plt.pause(0.00001)
        if ax == 500:
            plt.show()









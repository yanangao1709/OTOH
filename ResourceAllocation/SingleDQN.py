import torch
import random
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

MEMORY_CAPACITY = 2000
NUM_STATES = 18
request_num = 5
NUM_ACTIONS = 3
LR = 0.005
EPSILON = 0.9
BATCH_SIZE = 8
Q_NETWORK_ITERATION = 100
GAMMA = 0.9
EPISODES = 1000

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.set_seed(1)

        self.input_layer = nn.Linear(NUM_STATES, 32)
        self.input_layer.weight.data.normal_(0, 0.1)

        self.hidden_layer1 = nn.Linear(32,64)
        self.hidden_layer1.weight.data.normal_(0, 0.1)
        self.hidden_layer2 = nn.Linear(64,32)
        self.hidden_layer2.weight.data.normal_(0, 0.1)

        self.req_layers = {}
        for r in range(request_num):
            r_layer = nn.Linear(32, 32)
            r_layer.weight.data.normal_(0, 0.1)
            r_candroute_layer = nn.Linear(32, NUM_ACTIONS)
            r_candroute_layer.weight.data.normal_(0, 0.1)
            self.req_layers[r] = [r_layer, r_candroute_layer]

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        v = []
        for r in range(request_num):
            x = F.relu(self.req_layers[r][0](x))
            v.append(F.relu(self.req_layers[r][1](x)))
        return tuple(v)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

class DQN:
    def __init__(self):
        self.memory_counter = 0
        self.fig, self.ax = plt.subplots()
        self.reward_decomposer = None

        self.eval_net = Net()
        self.target_net = Net()
        # storage data of state, action ,reward and next state
        self.memory = np.zeros((MEMORY_CAPACITY, 42))

        self.learn_counter = 0
        self.optimizer = Adam(self.eval_net.parameters(), LR)
        self.loss = nn.MSELoss()

    def choose_action(self, state):
        # notation that the function return the action's index nor the real action
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        action = []
        if np.random.randn() <= EPSILON:
            action_value = self.eval_net.forward(state)
            for av in action_value:
                a = torch.max(av, 1)[1].data.item()
                action.append(a)
        else:
            for i in range(request_num):
                a = np.random.randint(0, NUM_ACTIONS)
                action.append(a)
        return action

    def store_trans(self, state, action, reward, next_state):
        if self.memory_counter % 500 == 0:
            print("The experience pool collects {} time experience".format(self.memory_counter))
        index = self.memory_counter % MEMORY_CAPACITY
        trans = np.hstack((state, action, reward, next_state))
        self.memory[index,] = trans
        self.memory_counter += 1

    def learn(self):
        # sample data
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memorie = self.memory[sample_index, :]
        # train the agent
        self.train(batch_memorie)
        # train the decomposer
        # decompose.train_decomposer(self.reward_decomposer, batch_memories, self.reward_optimiser)

    def train(self, EpisodeBatch):
        # copy parameters to target each 100 episodes

        if self.learn_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter += 1

        batch_state = torch.FloatTensor(EpisodeBatch[:, :NUM_STATES])
        batch_action = torch.LongTensor(EpisodeBatch[:, NUM_STATES:NUM_STATES + request_num].astype(int))
        batch_next_state = torch.FloatTensor(EpisodeBatch[:, -NUM_STATES:])
        batch_rewards = torch.FloatTensor(EpisodeBatch[:,NUM_STATES + request_num:NUM_STATES + request_num+1])

        q_eval = torch.reshape(torch.stack(self.eval_net(batch_state)), [BATCH_SIZE, request_num, NUM_ACTIONS]).gather(2, torch.reshape(batch_action, [BATCH_SIZE,tohp.request_num,1]))
        q_next = torch.reshape(torch.stack(self.target_net(batch_next_state)), [BATCH_SIZE, request_num, NUM_ACTIONS])
        q_next_max = torch.reshape(torch.max(q_next, 2).values, [BATCH_SIZE, request_num, 1])
        q_target = q_next_max.clone()
        for r in range(request_num):
            # q_target[:,r,:] = batch_rewards[:,:,i] + RLhp.GAMMA * q_next[:,r,:]
            q_target[:,r,:] = batch_rewards + GAMMA * q_next_max[:,r, :]

        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()


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

from QuantumEnv import RequestAndRouteGeneration as rrg
from TOQN import TOQNHyperparameters as tohp

H_RKN = [
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
]
route_num = 5
node_num = 18
HOPS = [4,4,4,4,5]


class QuantumNetwork:
    def __init__(self):
        self.requests = None
        self.selectedRoutes = None
        self.agent_local_env = []
        self.node_cap = None
        self.H_RKN = []   # r 请求的 k路径 有没有经过这个点

        self.num_steps = 5

    def obtain_requests(self):
        rg = rrg.RequestAndRouteGeneration()
        requests = rg.request_routes_generation()
        return requests

    def obtain_H_RKN(self):
        for r in range(tohp.request_num):
            k_pos = []
            for k in range(tohp.candidate_route_num):
                pos = []
                for m in range(tohp.nodes_num):
                    route = self.requests[r].getCandidateRoutes()
                    if m+1 in route[k]:
                        pos.append(1)
                    else:
                        pos.append(0)
                k_pos.append(pos)
            self.H_RKN.append(k_pos)

    def get_H_RKN(self):
        return self.H_RKN

    def reset(self):
        if self.requests:
            self.requests.clear()
        # self.requests = self.obtain_requests()
        self.requests = [[1,11], [2,13], [3,14], [2,9], [7,18]]
        # self.obtain_H_RKN()
        self.node_cap = [2,3,3,4,6,4,2,5,1,3,7,3,2,4,2,4,1,2]
        # random photon allocation
        photonallocated = [
            [2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 3, 2, 2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 2, 0, 2]
        ]
        state = None
        return state, photonallocated

    def setSelectedRoutes(self, selectedroutes):
        self.selectedRoutes = selectedroutes

    def transformStates(self, states):
        return self.get_states()

    def get_states(self):
        state = []
        # for r in range(route_num):
        #     state.append(self.requests[r][0])
        #     state.append(self.requests[r][1])
        for v in range(node_num):
            state.append(self.node_cap[v])
        return state

    def calculate_reward(self, actions):
        reward = 0
        for i in range(route_num):
            totalPho = 0
            for j in range(node_num):
                totalPho += H_RKN[i][j] * actions[j][i]
            r_thr = totalPho/HOPS[i]
            reward += r_thr
        return reward

    def transmit(self, action):
        global_reward = 0
        # 先状态迁移
        for j in range(node_num):
            for i in range(route_num):
                if H_RKN[i][j] == 1 and self.node_cap[j] > 0:
                    self.node_cap[j] -= action[i]
                    global_reward += action[i]

        reward = sum(action) * 100
        for j in range(node_num):
            if self.node_cap[j] < 0:
                reward -= 300 #* abs(self.node_cap[j])
        return self.get_states(), reward, global_reward


    def step(self, action, step_counter):
        next_state, reward, global_reward = self.transmit(action)
        # 判断是否结束
        done = self.check_termination(step_counter)
        return next_state, reward, global_reward, done

    def generateRequestsandRoutes(self):
        rg = rrg.RequestAndRouteGeneration()
        self.requests = rg.request_routes_generation()

    def getEngState(self, i, i_cp, j, j_cp):
        state_probs = self.multi_qubit_entgle.redefine_assign_qstate_of_multiqubits(i, i_cp, j, j_cp)
        return state_probs

    def check_termination(self, step_counter):
        if step_counter > self.num_steps:
            return True
        else:
            return False


if __name__ == '__main__':
    # opr = OptimalRS()
    # ps = StoragePolicy()

    env = QuantumNetwork()
    agents = DQN()
    acc_reward = []
    axx = []
    for episode in range(EPISODES):
        state, photonallocated = env.reset()
        step_counter = 0
        total_reward = 0
        while True:
            step_counter += 1
            # route selection
            # opr.set_photon_allocation(photonallocated)
            # selected_route = opr.get_route_from_CRR(episode, ps)
            selected_route = [[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
            # ps.storage_policy(opr.get_Y(), photonallocated, episode)
            env.setSelectedRoutes(selected_route)
            state = env.transformStates(state)
            # resource allocation
            action = agents.choose_action(state)
            next_state, reward, global_reward, done = env.step(action, step_counter)
            agents.store_trans(state, action, reward, next_state)
            total_reward += reward
            if done:
                break
            if agents.memory_counter >= MEMORY_CAPACITY:
                agents.learn()
            state = next_state
            # photonallocated = agents.get_PApolicy(action)
            # print("------Step_counter is " + str(step_counter))
        print(episode)
        acc_reward.append(total_reward / step_counter)  # total_reward/step_counter
        axx.append(episode)
        agents.plot(agents.ax, acc_reward)
        # print("In the" + str(episode) + " times transmission, the total throughput of reqSet R is " + str(total_reward))
    plt.xlabel("episodes")
    plt.ylabel("throughput")
    plt.plot(axx, acc_reward, 'b-')
    plt.show()
    res = {"x": axx, "acc_reward": acc_reward}
    pd.DataFrame(res).to_csv('./Results/throughput.csv', index=False)









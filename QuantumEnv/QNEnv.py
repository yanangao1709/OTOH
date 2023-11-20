# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 04-08-2023                                      #
#      Goals: implement the quantum network environment       #
#             for request response                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from QuantumEnv import RequestAndRouteGeneration as rrg
from TOQN import TOQNHyperparameters as tohp
from Topology import TOQNTopology as toTop
from Common.Throughput import Thr

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
        states = None
        return states, photonallocated

    def setSelectedRoutes(self, selectedroutes):
        self.selectedRoutes = selectedroutes

    def transformStates(self, states):
        return self.get_states()

    def get_states(self):
        # states = {}
        # for m in range(tohp.nodes_num):
        #     state = []
        #     for r in range(tohp.request_num):
        #         state.append(self.requests[r].getSource())
        #         state.append(self.requests[r].getDestination())
        #         state.append(self.requests[r].getVolumn())
        #         r_canRoutes = self.requests[r].getCandidateRoutes()
        #         if m + 1 in r_canRoutes[self.selectedRoutes[r].index(1)]:
        #             state.append(1)
        #         else:
        #             state.append(0)
        #     for v in range(tohp.nodes_num):
        #         if toTop.LINK_LENS[m][v]:
        #             state.append(1)
        #         else:
        #             state.append(0)
        #     state.append(self.node_cap[m])
        #     states[m] = state
        # return states

        states = {}
        for m in range(node_num):
            state = []
            # for r in range(route_num):
            #     state.append(self.requests[r][0])
            #     state.append(self.requests[r][1])
            for v in range(node_num):
                state.append(self.node_cap[v])
            states[m] = state
        return states

    def calculate_reward(self, actions):
        reward = 0
        for i in range(route_num):
            totalPho = 0
            for j in range(node_num):
                totalPho += H_RKN[i][j] * actions[j][i]
            r_thr = totalPho/HOPS[i]
            reward += r_thr
        return reward

    def transmit(self, actions):
        # reward = self.calculate_reward(actions) * 100
        reward = sum(actions) * 100
        # 先状态迁移
        for i in range(route_num):
            for j in range(node_num):
                if H_RKN[i][j] == 1:
                    self.node_cap[j] -= actions[j][i]
        # 判断约束
        for j in range(node_num):
            if self.node_cap[j] < 0:
                reward -= 10

        return self.get_states(), reward

    def step(self, actions, step_counter):
        next_states, reward = self.transmit(actions)
        # 判断是否结束
        done = self.check_termination(step_counter)
        return next_states, reward, done

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









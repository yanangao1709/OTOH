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


class QuantumNetwork:
    def __init__(self):
        self.requests = None
        self.selectedRoutes = None
        self.agent_local_env = []
        self.node_remain_cap = None
        self.H_RKN = []   # r 请求的 k路径 有没有经过这个点

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
        self.requests = self.obtain_requests()
        self.obtain_H_RKN()
        self.node_remain_cap = toTop.NODE_CPA
        # random photon allocation
        photonallocated = [
            [2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 6, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 3, 2, 2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 2, 8, 2]
        ]
        states = None
        return states, photonallocated

    def setSelectedRoutes(self, selectedroutes):
        self.selectedRoutes = selectedroutes

    def transformStates(self, states):
        return self.get_states()

    def get_states(self):
        states = {}
        for m in range(tohp.nodes_num):
            state = []
            for r in range(tohp.request_num):
                state.append(self.requests[r].getSource())
                state.append(self.requests[r].getDestination())
                state.append(self.requests[r].getVolumn())
                r_canRoutes = self.requests[r].getCandidateRoutes()
                if m + 1 in r_canRoutes[self.selectedRoutes[r].index(1)]:
                    state.append(1)
                else:
                    state.append(0)
            for v in range(tohp.nodes_num):
                if toTop.LINK_LENS[m][v]:
                    state.append(1)
                else:
                    state.append(0)
            state.append(self.node_remain_cap[m])
            states[m] = state
        return states

    def step(self, actions):
        for m in range(tohp.nodes_num):
            for r in range(tohp.request_num):
                if self.node_remain_cap[m] - actions[m][r] >= 0:
                    continue
                r_canRoutes = self.requests[r].getCandidateRoutes()
                if m+1 in r_canRoutes[self.selectedRoutes[r].index(1)]:
                    self.node_remain_cap[m] -= actions[m][r]
        next_states = self.get_states()
        throughput = Thr()
        reward = throughput.get_throuthput(self.requests, self.selectedRoutes, actions, self.H_RKN)
        return next_states, reward

    def generateRequestsandRoutes(self):
        rg = rrg.RequestAndRouteGeneration()
        self.requests = rg.request_routes_generation()

    def getEngState(self, i, i_cp, j, j_cp):
        state_probs = self.multi_qubit_entgle.redefine_assign_qstate_of_multiqubits(i, i_cp, j, j_cp)
        return state_probs

    def get_fidelity_of_route(self, r, k):
        # ij 一段一段的
        # 一段一段的，量子状态概率
        fidelity = 0
        route = r.candidates[k]
        for i in range(len(route)):
            link_qs = self.getEngState(i,)









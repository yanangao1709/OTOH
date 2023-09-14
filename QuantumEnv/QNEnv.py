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

    def obtain_requests(self):
        rg = rrg.RequestAndRouteGeneration()
        requests = rg.request_routes_generation()
        return requests

    def reset(self):
        if self.requests:
            self.requests.clear()
        self.requests = self.obtain_requests()
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
        states = {}
        for m in range(tohp.nodes_num):
            state = []
            for r in range(tohp.request_num):
                state.append(self.requests[r].getSource())
                state.append(self.requests[r].getDestination())
                state.append(self.requests[r].getVolumn())
                r_canroutes = self.requests[r].getCandidateRoutes()
                if m+1 in r_canroutes[self.selectedRoutes[r].index(1)]:
                    state.append(1)
                else:
                    state.append(0)
            for v in range(tohp.nodes_num):
                if toTop.LINK_LENS[m][n]:
                    state.append(1)
                else:
                    state.append(0)
            state.append(toTop.NODE_CPA[m])
        states[m] = state
        return states

    def step(self, actions):
        thr = Thr()
        test = 1

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
        test = 1









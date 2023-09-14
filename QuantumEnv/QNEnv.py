# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 04-08-2023                                      #
#      Goals: implement the quantum network environment       #
#             for request response                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from QuantumEnv import RequestAndRouteGeneration as rrg
from TOQN import TOQNHyperparameters as tohp
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
        states = None
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
        return states, photonallocated

    def transformStates(self, states):
        # 由原先的状态信息，转换为已知选择路径的信息

        test = 1

    def step(self, actions):
        thr = Thr()
        test = 1

    def setSelectedRoutes(self, selectedroutes):
        self.selectedRoutes = selectedroutes

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









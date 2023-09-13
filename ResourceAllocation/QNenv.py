# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 04-08-2023                                      #
#      Goals: implement the quantum network environment       #
#             for request response                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from Topology import RequestAndRouteGeneration as rrg
from QuantumState.QuantumNode import MultiqubitsEntanglement as mqe
from TOQN import TOQNHyperparameters as tohp


class QuantumNetwork:
    def __init__(self):
        self.requests = self.obtain_requests()
        self.sr = None
        self.agent_local_env = []

    def obtain_requests(self):
        rg = rrg.RequestAndRouteGeneration()
        requests = rg.request_routes_generation()
        return requests

    def reset(self):
        for m in range(tohp.topology_myself_nodes_num):
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









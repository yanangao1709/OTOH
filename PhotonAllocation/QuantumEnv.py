# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 04-08-2023                                      #
#      Goals: implement the quantum network environment       #
#             for request response                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from Topology import RequestAndRouteGeneration as rrg
from QuantumState.QuantumNode import MultiqubitsEntanglement as mqe


class QuantumEnv:
    def __init__(self):
        self.requests = self.generateRequests() # env初始化即可生成请求
        self.multi_qubit_entgle = mqe()


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

    def getRequetsandCandidateRoutes(self):
        rg = rrg.RequestAndRouteGeneration()
        requests = rg.request_routes_generation()








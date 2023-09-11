# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: Obtain the optimal route for                    #
#             each request                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from gurobipy import *
import TOQNHyperparameters as tohp
from Topology.TOQNTopology import H_RKN, HOPS
from RouteSelection.Constraints import Constraints

# one time slot one route selection
# with known photon allocation policy M_m^{r,t}

class OptimalRS:
    def __init__(self, M):
        self.request_num = tohp.request_num
        self.candidate_route_num = tohp.candidate_route_num
        self.node_num = tohp.topology_myself_nodes_num
        self.T_thr = tohp.T_thr
        self.M = M
        self.acc_throughput = 0

    def addVar(self, m):
        y_vars = m.addVars(self.request_num * self.candidate_route_num, vtype=GRB.BINARY)
        Y_vars = []
        for i in range(self.request_num):
            Y_temp = []
            for j in range(self.candidate_route_num):
                Y_temp.append(y_vars[i * self.candidate_route_num + j])
            Y_vars.append(Y_temp)
        return Y_vars

    def obtain_request_throughput(self):
        req_thp = []
        for i in range(self.request_num):
            req = []
            for j in range(self.candidate_route_num):
                total_memory = 0
                for m in range(self.node_num):
                    total_memory += H_RKN[i][j][k] * self.M[i][m]
                req.append(total_memory/HOPS[j])
            req_thp.append(req)
        return req_thp

    def obtain_optimal_route(self, t):
        try:
            # 定义问题
            m = Model("IntegerProblem")
            # 定义变量
            Y_vars = self.addVar(m)

            # 定义目标函数
            obj = quicksum(Y_vars[r][k] * quicksum(H_RKN[r][k][i] * self.M[r][i]
                                                 for i in range(self.node_num)
                                                 ) / HOPS[r][k]
                         for r in range(self.request_num)
                         for k in range(self.candidate_route_num)
                         )
            m.setObjective(obj, GRB.MAXIMIZE)
            # define constraints
            # fidelity
            for r in range(self.request_num):
                m.addConstrs(
                    quicksum(Y_vars[r][k] * Constraints.obatin_fidelity(r, k, t)
                             for k in range(self.candidate_route_num)
                             ) >= tohp.F_thr, GRB.MINIMIZE
                )
            # delay
            for r in range(self.request_num):
                m.addConstrs(

                )
            # node_capacity


            m.optimize()
            print('Optimal solution', end=" ")
            for i in m.getVars():
                print('%s = %g' % (i.varName, i.x), end=" ")

            # # fidelity
            # m.addConstr((3 / 250) * y11 >= F_thr)
            # m.addConstr((0.1639) * y12 >= F_thr)
            # m.addConstr((48 / 700) * y21 >= F_thr)
            # m.addConstr((159 / 1400) * y22 >= F_thr)
            # # node capacity
            # m.addConstr(2 * y11 + 2 * y12 <= s1_capacity)
            # m.addConstr(3 * y11 + 3 * y12 <= d1_capacity)
            # m.addConstr(2 * y21 + 2 * y22 <= s2_capacity)
            # m.addConstr(2 * y21 + 2 * y22 <= d2_capacity)
            # m.addConstr(4 * y11 + 4 * y12 + 2 * y21 + 2 * y22 <= r1_capacity)
            # m.addConstr(5 * y11 + 5 * y12 + 3 * y21 + 3 * y22 <= r2_capacity)
            # # route selection
            # m.addConstr(y11 + y12 >= 1)
            # m.addConstr(y11 + y12 <= 1)
            # m.addConstr(y21 + y22 >= 1)
            # m.addConstr(y21 + y22 <= 1)
            #
            # m.write("IntegerProblem.lp")
            #
            # m.optimize()
            #
            # print('Optimal solution', end=" ")
            # for i in m.getVars():
            #     print('%s = %g' % (i.varName, i.x), end=" ")

        except GurobiError as e:
            print('Error code' + str(e.errno) + ":" + str(e))

        except AttributeError:
            print('Encountered an attribute error')

    def accumulate_throughput(self):
        self.acc_throughput.append()


if __name__ == '__main__':
    photonallocated = [
        [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
        [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
        [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    ]
    ors = OptimalRS(photonallocated)
    ors.obtain_optimal_route()
    print(2)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 28-07-2023                                      #
#      Goals: obtain the global optimal solution              #
#             of TOQN problem                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from gurobipy import *
from Topology import HyperParameters as thp
from Topology import RequestAndRouteGeneration as rrg


class TOQN:
    def __init__(self):
        self.request_num = thp.request_num
        self.candidate_route_num = thp.candidate_route_num
        self.node_num = thp.topology_myself_nodes_num
        self.T_thr = thp.T_thr

    def getRequetsandCandidateRoutes(self):
        rg = rrg.RequestAndRouteGeneration()
        requests = rg.request_routes_generation()
        candidate_routes = rg.route_generation(requests)
        return requests, candidate_routes

    def addVar(self, m):
        y_vars = m.addVars(self.request_num*self.candidate_route_num, vtype=GRB.BINARY)
        x_vars = m.addVars(self.request_num*self.node_num, vtype=GRB.INTEGER)
        Y_vars = []
        for i in range(self.request_num):
            Y_temp = []
            for j in range(self.candidate_route_num):
                Y_temp.append(y_vars[i*self.request_num+j])
            Y_vars.append(Y_temp)
        X_vars = []
        for i in range(self.request_num):
            X_temp = []
            for j in range(self.node_num):
                X_temp.append(x_vars[i*self.request_num+j])
            X_vars.append(X_temp)
        return Y_vars, X_vars

    def objective(self, m, y_var, x_var):
        # 路径信息
        # fidelity 计算
        # delay计算
        # capacity约束
        # 变量加和约束
        # m.setObjective(, GRB.MAXIMIZE)
        test = 1

    def getFidelity(self, r, k, ):
        fidelity = 0
        # obtain the quantum state

        # obtain the route information
        rg = rrg.RequestAndRouteGeneration()

    def addConstraints(self, m):
        for i in range(self.request_num):
            for j in range(self.candidate_route_num):
                m.addConstr(Y_vars[i][j] * self.getFidelity(i,j) * )


    def obtainGlobalOptimal(self):
        try:
            m = Model("IntegerProblem")
            Y_vars, X_vars = self.addVar(m)

            obj = quicksum(Y_vars[r][k][t] * X_vars[r][i]
                                    for r in range(self.request_num)
                                    for k in range(self.candidate_route_num)
                                    for t in range(self.T_thr)
                                    for i in range(self.node_num))
            m.setObjective(obj, GRB.MAXIMIZE)



            # fidelity_cons

            m.addConstr(, GRB.MINIMIZE)





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


if __name__ == '__main__':
    toqn = TOQN()
    toqn.obtainGlobalOptimal()
    requests, candidate_routes = toqn.getRequetsandCandidateRoutes()
    test = 1


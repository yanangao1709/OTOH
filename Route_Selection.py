import numpy as np
from gurobipy import *
import HyperParameters as hp
from gurobipy import GRB, quicksum as qsum

class RouteSelectionILP:
    def __init__(self, t, M_i_r):
        super(RouteSelectionILP, self).__init__()
        self.t = t
        self.node_capacities = hp.NODE_CAPACITY

        self.rho = 1

        self.M_i_r = M_i_r

        # time slot t = 0

    def calculate_delay(self, r, k):
        delay = 0
        for i in range(hp.TOPOLOGY_SCALE):
            for j in range(hp.TOPOLOGY_SCALE):
                if i == j:
                    continue
                if (self.M_i_r[i][r] + self.M_i_r[j][r]) == 0:
                    continue
                delay += hp.H_IJRK.H_IJRK[i][j][r][k] * (hp.EDGES[i][j])/(hp.GAMMA*(self.M_i_r[i][r] + self.M_i_r[j][r]))
        return delay

    def integerRun(self):
        selected_route = []
        try:
            A = np.ones([hp.REQUEST_NUM,hp.CANDIDATE_ROUTE_NUM])
            m = Model("IntegerProblem")
            variable_Y = m.addVars(hp.REQUEST_NUM, hp.CANDIDATE_ROUTE_NUM, vtype=GRB.BINARY)
            for r in range(hp.REQUEST_NUM):
                for k in range(hp.CANDIDATE_ROUTE_NUM):
                    A[r][k] = self.calculate_delay(r,k)

            m.setObjective(qsum(A[i][j] * variable_Y[i, j] for i in range(hp.REQUEST_NUM) \
                                for j in range(hp.CANDIDATE_ROUTE_NUM)), GRB.MINIMIZE)

            # fidelity
            # m.addConstr((3/250)*y11 >= F_THR)
            # m.addConstr((0.1639)*y12 >= F_THR)
            # m.addConstr((48/700)*y21 >= F_THR)
            # m.addConstr((159/1400)*y22 >= F_THR)

            # node capacity constraint
            # for i in range(hp.TOPOLOGY_SCALE):
            #     node_total_used_capacity = 0
            #     for j in range(hp.TOPOLOGY_SCALE):
            #         if i == j:
            #             continue
            #         for r in range(hp.REQUEST_NUM):
            #             for k in range(hp.CANDIDATE_ROUTE_NUM):
            #                 if hp.H_IJRK.H_IJRK[i][j][r][k] == 1:
            #                     node_total_used_capacity += self.M_i_r[i][r] * self.variable_y[r][k]
            #     m.addConstr(node_total_used_capacity <= self.node_capacities[i])

            # route selection
            # for r in range(hp.REQUEST_NUM):
            #     r_route_num = 0
            #     for k in range(hp.CANDIDATE_ROUTE_NUM):
            #         r_route_num += self.variable_y[r][k]
            #     m.addConstr(r_route_num >= 1)

            m.write(".\\models\\IntegerProblem.lp")

            m.optimize()

            # print('Optimal solution', end=" ")
            for i in m.getVars():
                # print('%s = %g' % (i.varName, i.x), end=" ")
                selected_route.append(i.x)

        except GurobiError as e:
            # print('Error code' + str(e.errno) + ":" + str(e))
            test = 1

        except AttributeError:
            # print('Encountered an attribute error')
            test = 1

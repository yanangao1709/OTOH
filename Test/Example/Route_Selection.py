import numpy as np
from gurobipy import *
from Test.Example import HyperParameters as hp
from gurobipy import GRB, quicksum as qsum
import random

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
                if (self.M_i_r[r][i] + self.M_i_r[r][j]) == 0:
                    continue
                delay += Test.Example.H_IJRK[i][j][r][k] * (hp.EDGES[i][j]) / (hp.GAMMA * (self.M_i_r[r][i] + self.M_i_r[r][j]))
        return delay

    def integerRun(self):
        selected_routes = []
        try:
            A = np.ones([hp.REQUEST_NUM,hp.CANDIDATE_ROUTE_NUM])
            m = Model("IntegerProblem")
            variable_Y = m.addVars(hp.REQUEST_NUM, hp.CANDIDATE_ROUTE_NUM, vtype=GRB.BINARY)
            # delay
            for r in range(hp.REQUEST_NUM):
                for k in range(hp.CANDIDATE_ROUTE_NUM):
                    A[r][k] = self.calculate_delay(r,k)

            # throughput
            # for r in range(hp.REQUEST_NUM):
            #     for k in range(hp.CANDIDATE_ROUTE_NUM):
            #         route_information = hp.REQUESTS_ROUTES[r][k]
            #         qubits_allocated = []
            #         for a in range(hp.AGENT_NUM):
            #             if a in route_information:
            #                 qubits_allocated.append(self.M_i_r[r][a])
            #
            #         # A[r][k] = (sum(qubits_allocated))/len(qubits_allocated)
            #         A[r][k] = sum(qubits_allocated)


            m.setObjective(qsum(A[i][j] * variable_Y[i, j] for i in range(hp.REQUEST_NUM) \
                                for j in range(hp.CANDIDATE_ROUTE_NUM)), GRB.MAXIMIZE)

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
            #                 if hp.H_IJRK[i][j][r][k] == 1:
            #                     node_total_used_capacity += self.M_i_r[r][i] * variable_Y[r,k]
            #     m.addConstr(node_total_used_capacity <= self.node_capacities[i], name="node capacity")

            # route selection
            for r in range(hp.REQUEST_NUM):
                r_route_num = 0
                for k in range(hp.CANDIDATE_ROUTE_NUM):
                    r_route_num += variable_Y[r,k]
                m.addConstr(r_route_num >= 1, name="route-selection")

            m.write(".\\models\\IntegerProblem.lp")

            m.optimize()

            # print('Optimal solution', end=" ")
            results = []
            for i in m.getVars():
                # print('%s = %g' % (i.varName, i.x), end=" ")
                results.append(int(i.x))
            for r in range(hp.REQUEST_NUM):
                for cr in range(hp.CANDIDATE_ROUTE_NUM):
                    if results[r*3+cr] == 1:
                        selected_routes.append(cr)
                        break
            return selected_routes, m.getObjective().getValue()

        except GurobiError as e:
            selected_routes = [random.randint(0, hp.CANDIDATE_ROUTE_NUM - 1) for r in range(hp.REQUEST_NUM)]
            print('Error code' + str(e.errno) + ":" + str(e))
            return selected_routes, m.getObjective().getValue()

        except AttributeError:
            selected_routes = [random.randint(0, hp.CANDIDATE_ROUTE_NUM - 1) for r in range(hp.REQUEST_NUM)]
            print('Encountered an attribute error')
            return selected_routes, m.getObjective().getValue()

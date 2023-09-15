# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: Obtain the optimal route for                    #
#             each request                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import random
from gurobipy import *
from TOQN import TOQNHyperparameters as tohp
from Topology.TOQNTopology import H_RKN, HOPS, ROUTE_LEN, NODE_CPA
from RouteSelection.Constraints import Constraints

# one time slot one route selection
# with known photon allocation policy M_m^{r,t}

class OptimalRS:
    def __init__(self):
        self.request_num = tohp.request_num
        self.candidate_route_num = tohp.candidate_route_num
        self.node_num = tohp.nodes_num
        self.T_thr = tohp.T_thr
        self.acc_throughput = 0
        self.Y = None
        self.M = None
        self.Flag = True

    def set_photon_allocation(self, M):
        self.M = M

    def addVar(self, m):
        y_vars = m.addVars(self.request_num * self.candidate_route_num, vtype=GRB.CONTINUOUS)
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

    def transformY(self, sol):
        Y = []
        for r in range(tohp.request_num):
            yy = []
            prob_r = sol[tohp.candidate_route_num*r : tohp.candidate_route_num*r+3]
            rel_y = random.choices([0,1,2], prob_r)[0]
            for k in range(tohp.candidate_route_num):
                if k == rel_y:
                    yy.append(1)
                else:
                    yy.append(0)
            Y.append(yy)
        return Y

    def storage(self, Y, M, t):
        ps = StoragePolicy()
        ps.storage_policy(Y, M, t)

    def obtain_optimal_route(self, t, ps):
        cons = Constraints(ps)
        try:
            # define problem
            m = Model("LinearProblem")
            # define variables
            Y_vars = self.addVar(m)

            # define objective
            obj = quicksum(Y_vars[r][k] * quicksum(H_RKN[r][k][i] * self.M[r][i]
                                                 for i in range(self.node_num)
                                                 ) / HOPS[r][k]
                         for r in range(self.request_num)
                         for k in range(self.candidate_route_num)
                         )
            m.setObjective(obj, GRB.MAXIMIZE)
            # define constraints
            # fidelity
            m.addConstrs(
                quicksum(
                    Y_vars[r][k] * cons.obatin_fidelity(r, k, self.M, t)
                         for k in range(self.candidate_route_num)
                         ) >= tohp.F_thr
                for r in range(self.request_num)
            )
            # delay
            m.addConstrs(
                quicksum(
                    Y_vars[r][k] * ROUTE_LEN[r][k] + cons.obtain_delay(r, t)
                    for k in range(self.candidate_route_num)
                ) <= tohp.D_thr
                for r in range(self.request_num)
            )
            # node_capacity
            m.addConstrs(
                quicksum(
                    Y_vars[r][k] * H_RKN[r][k][m] * self.M[r][m] + cons.obtain_node_cap(m)
                    for r in range(self.request_num)
                    for k in range(self.candidate_route_num)
                ) <= NODE_CPA[m]
                for m in range(self.node_num)
            )
            # route selection
            m.addConstrs(
                quicksum(Y_vars[r][k]
                         for k in range(self.candidate_route_num)
                         ) >= 1
                for r in range(self.request_num)
            )
            m.addConstrs(
                quicksum(Y_vars[r][k]
                         for k in range(self.candidate_route_num)
                         ) <= 1
                for r in range(self.request_num)
            )
            m.write("RouteSelection-Linear.lp")
            m.optimize()
            print('Optimal solution', end=" ")
            sol =  []
            for i in m.getVars():
                sol.append(i.x)
                print('%s = %g' % (i.varName, i.x), end=" ")
            self.Y = self.transformY(sol)
        except GurobiError as e:
            self.Flag = False
            print('Error code' + str(e.errno) + ":" + str(e))
        except AttributeError:
            self.Flag = False
            print('Encountered an attribute error')

    def get_route_from_CRR(self, t, ps):
        if self.Flag:
            self.obtain_optimal_route(t, ps)
            return self.Y
        else:
            return ps.get_last_Y_policy()

    def get_Y(self):
        return self.Y

# ------test for OptimalRoute-------
# if __name__ == '__main__':
#     photonallocated = [
#         [2,2,2,2,4,2,2,2,2,2,2,6,2,2,2,2,2,2],
#         [2,2,2,2,2,2,4,2,2,2,2,2,2,2,2,2,2,2],
#         [2,2,2,2,2,2,2,2,2,2,6,2,2,2,2,2,2,2],
#         [2, 2, 2, 2, 2, 2, 2, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
#         [2, 2, 2, 3, 2, 2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 2, 8, 2]
#     ]
#     ors = OptimalRS()
#     ors.set_photon_allocation(photonallocated)
#     ors.obtain_optimal_route(0)
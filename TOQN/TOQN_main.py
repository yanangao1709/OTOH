# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 28-07-2023                                      #
#      Goals: obtain the global optimal solution              #
#             of TOQN problem                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from gurobipy import *
import TOQNHyperparameters as tohp
from QuantumEnv import HyperParameters as qshp
from QuantumEnv import RequestAndRouteGeneration as rrg
from Topology.TOQNTopology import ROUTES, LINK_LENS, HOPS, D_VOLUMN, H_RKN, ROUTE_LEN, NODE_CPA

class TOQNConstraints:
    def __init__(self, Y, X):
        self.Y = Y
        self.X = X
        self.delay_flag = False
        self.delay = 0

    def obtain_delay(self, r, t):
        if self.delay_flag:
            return self.delay
        transmitted_data = 0
        for i in range(t):
            for k in range(tohp.candidate_route_num):
                transmitted_data_k_r = 0
                for m in range(tohp.nodes_num):
                    transmitted_data_k_r += H_RKN[r][k][m] * self.X[r][m][t]
                transmitted_data += self.Y[r][k][t] * transmitted_data_k_r / HOPS[r][k]
            if transmitted_data >= D_VOLUMN[r]:
                self.delay = t
                self.delay_flag = True
        return self.delay

    def obatin_fidelity(self, r, k, t):
        route = ROUTES[r][k]
        H = HOPS[r][k]
        mulM = 1
        sumM = 0
        sumLink = 0
        for i in range(H):
            if i == 0:
                continue
            mulM *= self.X[r][route[i] - 1][t]
            sumM += self.X[r][route[i] - 1][t]
            sumLink += LINK_LENS[route[i - 1] - 1][route[i] - 1]
        F = (pow(qshp.p, H) * pow(qshp.d, H / 2) * mulM * pow((1 - qshp.p), (qshp.d * sumM - H))
             * pow(math.e, -1 * qshp.tau * sumLink * t))
        return F

    def obtain_node_cap(self, m, t):
        occupied_photons = 0
        for i in range(t):
            for r in range(tohp.request_num):
                for k in range(tohp.candidate_route_num):
                    if self.Y[r][k][t] == 1 and H_RKN[r][k][m] == 1:
                        occupied_photons += self.X[r][m][t]
        return occupied_photons


class TOQN:
    def __init__(self):
        self.request_num = tohp.request_num
        self.candidate_route_num = tohp.candidate_route_num
        self.node_num = tohp.topology_myself_nodes_num
        self.T_thr = tohp.T_thr

    def getRequetsandCandidateRoutes(self):
        rg = rrg.RequestAndRouteGeneration()
        requests = rg.request_routes_generation()
        candidate_routes = rg.route_generation(requests)
        return requests, candidate_routes

    def addVar(self, m):
        y_vars = m.addVars(self.request_num*self.candidate_route_num*self.T_thr, vtype=GRB.BINARY)
        x_vars = m.addVars(self.request_num*self.node_num*self.T_thr, vtype=GRB.INTEGER)
        Y_vars = []
        for i in range(self.request_num):
            Y_temp = []
            for j in range(self.candidate_route_num):
                Y_tt = []
                for t in range(tohp.T_thr):
                    Y_tt.append(y_vars[i*self.request_num+j*self.candidate_route_num+t])
                Y_temp.append(Y_tt)
            Y_vars.append(Y_temp)
        X_vars = []
        for i in range(self.request_num):
            X_temp = []
            for j in range(self.node_num):
                X_tt = []
                for t in range(tohp.T_thr):
                    X_tt.append(x_vars[i*self.request_num+j*self.node_num+t])
                X_temp.append(X_tt)
            X_vars.append(X_temp)
        return Y_vars, X_vars

    def obtainGlobalOptimal(self):
        try:
            # define the model
            m = Model("MixedIntegerNonLinearProblem")
            # define variables
            Y_vars, X_vars = self.addVar(m)
            cons = TOQNConstraints(Y_vars, X_vars)
            # define objective
            obj = quicksum(Y_vars[r][k][t] * quicksum(H_RKN[r][k][i] * X_vars[r][i][t]
                                                   for i in range(self.node_num)
                                                   ) / HOPS[r][k]
                           for r in range(self.request_num)
                           for k in range(self.candidate_route_num)
                           for t in range(self.T_thr)
                           )
            m.setObjective(obj, GRB.MAXIMIZE)
            # define constraints
            for t in range(self.T_thr):
                # fidelity
                m.addConstrs(
                    quicksum(
                        Y_vars[r][k][t] * cons.obatin_fidelity(r, k, t)
                        for k in range(self.candidate_route_num)
                    ) >= tohp.F_thr
                    for r in range(self.request_num)
                )
                # delay
                m.addConstrs(
                    quicksum(
                        Y_vars[r][k][t] * ROUTE_LEN[r][k] + cons.obtain_delay(r, t)
                        for k in range(self.candidate_route_num)
                    ) <= tohp.D_thr
                    for r in range(self.request_num)
                )
                # node_capacity
                m.addConstrs(
                    quicksum(
                        Y_vars[r][k][t] * H_RKN[r][k][m] * X_vars[r][m][t] + cons.obtain_node_cap(m, t)
                        for r in range(self.request_num)
                        for k in range(self.candidate_route_num)
                    ) <= NODE_CPA[m]
                    for m in range(self.node_num)
                )
                # route selection
                m.addConstrs(
                    quicksum(Y_vars[r][k][t]
                             for k in range(self.candidate_route_num)
                             ) >= 1
                    for r in range(self.request_num)
                )
                m.addConstrs(
                    quicksum(Y_vars[r][k][t]
                             for k in range(self.candidate_route_num)
                             ) <= 1
                    for r in range(self.request_num)
                )
            m.optimize()
            print('Optimal solution', end=" ")
            for i in m.getVars():
                print('%s = %g' % (i.varName, i.x), end=" ")
            m.optimize()
            print('Optimal solution', end=" ")
            for i in m.getVars():
                print('%s = %g' % (i.varName, i.x), end=" ")
        except GurobiError as e:
            print('Error code' + str(e.errno) + ":" + str(e))
        except AttributeError:
            print('Encountered an attribute error')

if __name__ == '__main__':
    toqn = TOQN()
    toqn.obtainGlobalOptimal()
    # requests, candidate_routes = toqn.getRequetsandCandidateRoutes()
    test = 1


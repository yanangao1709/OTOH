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
    def __init__(self, m, Y, X):
        self.Y = Y
        self.X = X
        self.delay_flag = m.addVar(vtype=GRB.BINARY)
        self.FalseVar = m.addVar(vtype = GRB.BINARY)
        m.addConstr(self.FalseVar == False)
        self.TrueVar = m.addVar(vtype=GRB.BINARY)
        m.addConstr(self.TrueVar == True)
        self.delay = m.addVar(vtype=GRB.INTEGER)

        self.eps = 0.0001
        self.M = 10 + self.eps

    def obtain_new_delay(self, m, r, t):
        sum_transmitted_data_last = 0
        for i in range(t):
            for k in range(tohp.candidate_route_num):
                sum_transmitted_dkr_last = 0
                for v in range(tohp.nodes_num):
                    sum_transmitted_dkr = m.addVar(vtype=GRB.CONTINUOUS, name='')
                    m.addConstr(sum_transmitted_dkr == sum_transmitted_dkr_last + H_RKN[r][k][v] * self.X[r][v][t])
                    sum_transmitted_dkr_last = sum_transmitted_dkr
                sum_transmitted_data = m.addVar(vtype=GRB.CONTINUOUS, name='')
                m.addConstr(
                    sum_transmitted_data == sum_transmitted_data_last + self.Y[r][k][t] * sum_transmitted_dkr / HOPS[r][
                        k])
                sum_transmitted_data_last = sum_transmitted_data
            volume = m.addVar(vtype=GRB.INTEGER)
            m.addConstr(volume == D_VOLUMN[r])
            tvar = m.addVar(vtype=GRB.INTEGER)
            m.addConstr(tvar == t)
            b = m.addVar(vtype=GRB.BINARY, name="b")
            m.addConstr(sum_transmitted_data >= volume + self.eps - self.M * (1 - b), name="bigM_constr1")
            m.addConstr(sum_transmitted_data <= volume + self.M * b, name="bigM_constr2")
            m.addConstr((b == 1) >> (self.delay == tvar), name="indicator_constr1")
            m.addConstr((b == 1) >> (self.delay_flag == self.TrueVar), name="indicator_constr1")
        return self.delay

    def obtain_delay(self, m, r, t):
        delay = m.addVar(vtype=GRB.INTEGER)

        b = m.addVar(vtype=GRB.BINARY, name="b")
        m.addConstr(self.delay_flag >= self.FalseVar + self.eps - self.M * (1 - b), name="bigM_constr1")
        m.addConstr(self.delay_flag <= self.FalseVar + self.M * b, name="bigM_constr2")
        m.addConstr((b == 1) >> (delay == self.delay), name="indicator_constr1")
        m.addConstr((b == 0) >> (delay == self.obtain_new_delay(m, r, t)), name="indicator_constr1")
        return delay

    def obtain_fidelity(self, m, r, k, t):
        route = ROUTES[r][k]
        H = HOPS[r][k]
        sumM_last = 0
        sumLink = 0
        mulM_last = 1
        for i in range(H):
            if i == 0:
                continue
            mulM = m.addVar(vtype=GRB.INTEGER, name='mulM')
            m.addConstr(mulM == mulM_last * self.X[r][route[i] - 1][t])
            mulM_last = mulM
            sumM = m.addVar(vtype=GRB.INTEGER, name='sumM')
            m.addConstr(sumM == sumM_last + self.X[r][route[i] - 1][t])
            sumM_last = sumM
            sumLink += LINK_LENS[route[i - 1] - 1][route[i] - 1]
        pow1 = m.addVar(vtype=GRB.CONTINUOUS, name='pow1')
        sumX = m.addVar(vtype=GRB.CONTINUOUS)
        m.addGenConstrExpA(sumX, pow1, 1 - qshp.p)
        m.addConstr(sumX == qshp.d * sumM - H)
        F = m.addVar(vtype=GRB.CONTINUOUS, name='fidelity')
        m.addConstr(F == (pow(qshp.p, H) * pow(qshp.d, H / 2) * mulM_last * pow1
             * pow(math.e, -1 * qshp.tau * sumLink * t)))
        return F

    def obtain_new_node_cap(self, m, v, i, r, k, b, sum_occupied_photons):
        if b and H_RKN[r][k][v] == 1:
            m.addConstr(sum_occupied_photons == self.X[r][v][i])
        else:
            m.addConstr(sum_occupied_photons == 0)
        return sum_occupied_photons

    def obtain_node_cap(self, m, v, t):
        b = m.addVar(vtype=GRB.BINARY, name="b")
        sum_occupied_photons_last = 0
        for i in range(t):
            for r in range(tohp.request_num):
                for k in range(tohp.candidate_route_num):
                    sum_occupied_photons = m.addVar(vtype=GRB.INTEGER, name='')
                    m.addConstr(self.Y[r][k][i] >= self.FalseVar + self.eps - self.M * (1 - b), name="bigM_constr1")
                    m.addConstr(self.Y[r][k][i] <= self.FalseVar + self.M * b, name="bigM_constr2")
                    m.addConstr((b == 1) >> (sum_occupied_photons_last == sum_occupied_photons_last + self.obtain_new_node_cap(m, v, i, r, k,True, sum_occupied_photons)))
                    m.addConstr((b == 0) >> (sum_occupied_photons_last == sum_occupied_photons_last + self.obtain_new_node_cap(m, v, i, r, k, False, sum_occupied_photons)))
        return sum_occupied_photons_last

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
            cons = TOQNConstraints(m, Y_vars, X_vars)
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
            # fidelity
            # m.addConstrs(
            #     quicksum(
            #         Y_vars[r][k][t] * cons.obtain_fidelity(m, r, k, t)
            #         for k in range(self.candidate_route_num)
            #     ) >= tohp.F_thr
            #     for r in range(self.request_num)
            #     for t in range(self.T_thr)
            # )

            # delay
            # m.addConstrs(
            #     quicksum(
            #         Y_vars[r][k][t] * ROUTE_LEN[r][k] + cons.obtain_delay(m, r, t)
            #         for k in range(self.candidate_route_num)
            #         for t in range(self.T_thr)
            #     ) <= tohp.D_thr
            #     for r in range(self.request_num)
            # )
            # node_capacity
            m.addConstrs(
                quicksum(
                    Y_vars[r][k][t] * H_RKN[r][k][v] * X_vars[r][v][t] + cons.obtain_node_cap(m, v, t)
                    for r in range(self.request_num)
                    for k in range(self.candidate_route_num)
                    for t in range(self.T_thr)
                ) <= NODE_CPA[v]
                for v in range(self.node_num)
            )
            # route selection
            # m.addConstrs(
            #     quicksum(Y_vars[r][k][t]
            #              for k in range(self.candidate_route_num)
            #              for t in range(self.T_thr)
            #              ) >= 1
            #     for r in range(self.request_num)
            # )
            # m.addConstrs(
            #     quicksum(Y_vars[r][k][t]
            #              for k in range(self.candidate_route_num)
            #              for t in range(self.T_thr)
            #              ) <= 1
            #     for r in range(self.request_num)
            # )
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


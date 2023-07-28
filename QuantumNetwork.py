# --------------------example------------------
# ACTION_SPACE = 3
# STATE_SPACE = 5
# SOURCE_NODE = 2
# DES_NODE = 4
# NODES = [1,2,3,4,5]
# NODES_CAPACITY = [5, 8, 14, 6, 3]
# REQUESTS = {1:[2,4],2:[5,3],3:[4,1]}
# PATH_SET = {1:[[2, 1, 4],[2, 5, 4],[2, 3, 4]],
#             2:[[5, 1, 3],[5, 4, 3],[5, 2, 3]],
#             3:[[4, 1],[4, 3, 1],[4, 5, 1]]
#             }
#
# LINK_LENGTH= [
#     [0,6,8,5,5],
#     [6,0,10,0,7],
#     [8,10,0,9,0],
#     [5,0,9,0,3],
#     [5,7,0,3,0]
# ]
# LENGTH = [11, 10, 17]
import math

import HyperParameters as hp

class Env:
    def __init__(self, Selected_routes):
        self.agent_num = hp.AGENT_NUM
        self.requests = hp.REQUESTS
        self.node_capacity = hp.NODE_CAPACITY
        self.selected_routes = Selected_routes
        self.before_fidelity = []
        self.accumulated_throughput = 0

    def reset(self, step_counter, selected_soutes):
        # node_capacity recover
        self.node_capacity = hp.NODE_CAPACITY
        self.before_fidelity = []
        self.accumulated_throughput = 0
        self.selected_routes = selected_soutes

        self.obss = []
        for agent in range(hp.AGENT_NUM):
            obs = []
            # request information
            obs += self.requests
            self.obss.append(obs)

        # route information
        for i in range(hp.TOPOLOGY_SCALE):
            route = selected_soutes[step_counter]
            route_information = hp.REQUESTS_ROUTES[step_counter][route]
            if i in route_information:
                self.obss[i].append(1)
            else:
                self.obss[i].append(0)

        # the neighbor nodes
        for i in range(hp.TOPOLOGY_SCALE):
            edge = hp.EDGES[i]
            for e in edge:
                if e > 0:
                    self.obss[i].append(1)
                else:
                    self.obss[i].append(0)
            self.obss[i].append(self.node_capacity[i])

        return self.obss

    def step(self, step_counter, actions):
        next_observations = self.transmit(step_counter, actions)
        global_reward = self.calculate_reward(step_counter, actions)
        done = self.check_termination(step_counter)
        return next_observations, global_reward, done

    def transmit(self, step_counter, actions):
        # node capacities changed
        for agent in range(hp.AGENT_NUM):
            act = actions[agent]
            route = self.selected_routes[step_counter]
            route_information = hp.REQUESTS_ROUTES[step_counter][route]
            if (agent in route_information) and (self.node_capacity[agent] > 0):
                self.node_capacity[agent] -= act

        next_obss = []
        for agent in range(hp.AGENT_NUM):
            # request information does not change temporally
            for rep in range(hp.AGENT_NUM):
                obs = []
                # request information
                obs += self.requests
                next_obss.append(obs)

        # route information does not change too temporally
        for i in range(hp.TOPOLOGY_SCALE):
            route = self.selected_routes[step_counter]
            route_information = hp.REQUESTS_ROUTES[step_counter][route]
            if i in route_information:
                next_obss[i].append(1)
            else:
                next_obss[i].append(0)

        # the neighbor nodes' capacities do change temporally with unchanged requests
        for agent in range(hp.AGENT_NUM):
            edge = hp.EDGES[agent]
            for e in edge:
                if e > 0:
                    next_obss[agent].append(self.node_capacity[e])
                else:
                    next_obss[agent].append(0)
            next_obss[agent].append(self.node_capacity[agent])

        return next_obss

    def calculate_delay(self, r, k, actions):
        delay = 0
        for i in range(hp.TOPOLOGY_SCALE):
            for j in range(hp.TOPOLOGY_SCALE):
                if i == j:
                    continue
                if (actions[i] + actions[j]) == 0:
                    continue
                delay += hp.H_IJRK[i][j][r][k] * (hp.EDGES[i][j])/(hp.GAMMA*(actions[i] + actions[j]))
        return delay

    def calculate_reward(self, step_counter, actions):
        # maximize sum_fidelity of an episode
        # step_counter用于fidelity的时序衰退，暂时还没考虑decoherence 和request 随时序变化
        # r_route_fidelity = 0
        # route_information = hp.REQUESTS_ROUTES[step_counter][self.selected_routes[step_counter]]
        # for l in range(len(route_information)-1):
        #     if hp.H_IJRK[route_information[l]-1][route_information[l+1]-1][step_counter][self.selected_routes[step_counter]] == 0:
        #         continue
        #     r_route_fidelity += (actions[route_information[l]-1]+actions[route_information[l+1]-1])\
        #                         /hp.EDGES[route_information[l]-1][route_information[l+1]-1]
        # self.before_fidelity.append(hp.GAMMA * r_route_fidelity)
        #
        # return sum(self.before_fidelity)

        # throughput
        qubits_allocated = 0
        route_information = hp.REQUESTS_ROUTES[step_counter][self.selected_routes[step_counter]]
        for l in range(len(route_information)-1):
            if hp.H_IJRK[route_information[l]-1][route_information[l+1]-1][step_counter][self.selected_routes[step_counter]] == 0:
                continue
            qubits_allocated += actions[route_information[l]-1]
        qubits_allocated += actions[route_information[l+1]-1]
        return 10*qubits_allocated

        # delay
        # delay = 0
        # for r in range(hp.REQUEST_NUM):
        #     delay += self.calculate_delay(step_counter, self.selected_routes[step_counter], actions)
        # return delay


    def capacity_eff(self):
        out_capacity_num = 0
        for agent in range(hp.AGENT_NUM):
            if self.node_capacity[agent] <= 0:
                out_capacity_num += 1
        if out_capacity_num > math.ceil(hp.XI * hp.AGENT_NUM):
            return True
        else:
            return False

    def check_termination(self, step_counter):
        # if step_counter >= hp.STEP_LIMITATION or self.capacity_eff():
        if step_counter >= hp.STEP_LIMITATION:
            return True
        else:
            return False


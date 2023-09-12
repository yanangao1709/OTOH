# Topology

TOPOLOGY_SCALE = 18
EDGES = [[0,4,3,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [4,0,0,6,2,0,0,0,0,0,0,0,0,0,5,0,0,0],
         [3,0,0,0,4,7,0,0,0,0,0,0,0,0,0,0,0,0],
         [8,0,0,0,5,0,0,6,4,0,0,0,0,0,0,0,0,0],
         [0,6,4,5,0,0,4,4,0,0,8,0,0,0,0,0,0,0],
         [0,2,7,0,0,0,3,0,0,0,0,0,0,2,3,0,0,0],
         [0,0,0,0,0,4,3,0,0,0,5,0,0,4,0,0,0,0],
         [0,0,0,6,4,0,0,0,2,0,0,0,0,0,0,0,0,11],
         [0,0,0,4,0,0,0,2,0,3,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,3,0,5,6,0,0,0,0,0,0],
         [0,0,0,0,8,0,5,0,0,5,0,4,3,0,0,5,0,0],
         [0,0,0,0,0,0,0,0,0,6,4,0,0,0,0,7,0,5],
         [0,0,0,0,0,0,0,0,0,0,3,0,0,6,0,4,0,0],
         [0,0,0,0,0,2,4,0,0,0,0,0,6,0,4,0,0,0],
         [0,5,0,0,0,3,0,0,0,0,0,0,0,4,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,5,7,4,0,0,0,8,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,9],
         [0,0,0,0,0,0,0,11,0,0,0,5,0,0,0,0,9,0]]

# NODE_CAPACITY = [2,3,3,4,6,4,2,5,1,3,7,3,2,4,2,4,1,2]
NODE_CAPACITY = [4, 9, 8, 4, 3, 6, 8, 4, 6, 5, 8, 2, 4, 8, 7, 9, 8, 7]

from Test.Example import H_IJRK

H_IJRK = H_IJRK.H_IJRK

# candidate route information
REQUESTS_ROUTES = [[[1,4,9,10,11],[1,4,5,11],[1,3,5,11]],
                   [[6,7,11,12],[6,14,7,11,12],[6,14,13,11,12]],
                   [[14,7,11,12,18],[14,13,16,17,18],[14,13,16,12,18]]]


# Quantum information
GAMMA = 0.09
TAU = 0.2
RHO = 1

# Problem
REQUESTS = [1,11,6,12,14,18]
REQUEST_NUM = 3
CANDIDATE_ROUTE_NUM = 3

# Global optimization
OPT_NUM = 10

# Decentralized RL
AGENT_NUM = TOPOLOGY_SCALE
LR = 0.001
STEP_LIMITATION = REQUEST_NUM
N_ACTIONS = 5
N_OBSERVATIONS = 26
BATCH_SIZE = 8
MEMORY_CAPACITY = 2000
EPISODES = 2000
EPSILON = 0.9
ROUTE_GENERATION_NUM = REQUEST_NUM
XI = 1/2   # CAPACITY OVER-USED RATE FOR EPISODE TERMINAL
RECORD_LENGTH = 54 # 28+3+1+28


# Centralized Randomized Rounding
F_THR = 0.000001
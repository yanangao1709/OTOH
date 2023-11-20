# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: RL Hyperparameters                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from TOQN import TOQNHyperparameters as tohp

# RL
EPISODES = 1000
MEMORY_CAPACITY = 200
X_thr = 2
NUM_STATES = 18
NUM_ACTIONS = X_thr + 1
LR = 0.005
EPSILON = 0.9

Q_NETWORK_ITERATION = 100
BATCH_SIZE = 8
GAMMA = 0.9
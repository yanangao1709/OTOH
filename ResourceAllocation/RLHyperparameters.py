# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: RL Hyperparameters                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from TOQN import TOQNHyperparameters as tohp
# Env


# RL
EPISODES = 1000
MEMORY_CAPACITY = 2000
X_thr = 2
NUM_STATES = 39
NUM_ACTIONS = X_thr + 1
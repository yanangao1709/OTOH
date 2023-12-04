# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: RL Hyperparameters                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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
lr_a = 3e-4
lr_c = 5e-4

# Reward decomposition variables
decompose_reward: True
reward_parameter_sharing: True
reward_batch_size: 100
reward_updates_per_batch: 100
reward_diff_threshold: 0.05
assume_binary_reward: False
reward_acc: 0.999
regularizing_weight: 0.00005
reward_index_in_obs: -1
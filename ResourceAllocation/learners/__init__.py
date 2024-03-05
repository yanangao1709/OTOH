from .q_learner import QLearner
REGISTRY = {}

# My algorithm lives under QLearner
REGISTRY["q_learner"] = QLearner
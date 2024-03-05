REGISTRY = {}

from .my_DQN import MyDQN

REGISTRY["myDQN"] = MyDQN

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent
from functools import partial
from ResourceAllocation.envs.multi_particle.multi_particle_env import MultiParticleEnv

# If we are using Traffic Environments
# from envs.traffic.traffic_env import TrafficEnv

# If we are using StarCraft Environments
# from smac.env import MultiAgentEnv, StarCraft2Env

from ResourceAllocation.envs.multiagentenv import MultiAgentEnv

import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}

REGISTRY["multi_particle"] = partial(env_fn, env=MultiParticleEnv)

# Only use if we have starcraft installed
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

# Only use if we have traffic SUMO installed
# REGISTRY["traffic"] = partial(env_fn, env=TrafficEnv)

# I think it doesnt go in here
# if sys.platform == "linux" or True:
#     os.environ.setdefault("SC2PATH",
#                           os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

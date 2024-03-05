import cv2
from ResourceAllocation.envs import REGISTRY as env_REGISTRY
from functools import partial  # 固定参数，形成一个新的函数
from ResourceAllocation.components.episode_buffer import EpisodeBatch
import numpy as np
import time
from QuantumEnv.QNEnv import QuantumNetwork as QN
from RouteSelection.OptimalRoute import OptimalRS
from Constraint.PolicyStorage import StoragePolicy


class EpisodeRunner:

    def __init__(self, args):
        self.args = args
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = QN()
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.n_episodes = 0

        self.delay_thr = 0

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def set_delay_thr(self, delay_thr):
        self.delay_thr = delay_thr

    def reset(self):
        self.batch = self.new_batch()
        photonallocated, selected_route = self.env.reset()
        self.n_episodes += 1
        self.t = 0
        self.opr = OptimalRS()
        self.ps = StoragePolicy(self.env.get_env_info()["episode_limit"])
        self.ps.storage_policy(selected_route, photonallocated, self.t)

        # set hyper parameters
        self.opr.set_delay_thr(self.delay_thr)
        return photonallocated

    def run(self, test_mode=False):
        photonallocated = self.reset()
        terminated = False
        episode_return = 0
        CRR_times = 0

        while not terminated:
            self.opr.set_photon_allocation(photonallocated)
            selected_route, flag = self.opr.get_route_from_CRR(self.t, self.ps)
            if flag:
                CRR_times += 1
            self.env.setSelectedRoutes(selected_route)

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            try:
                self.batch.update(pre_transition_data, ts=self.t)
            except Exception as e:
                print("PROBLEM IN UPDATE")
                print(e)
                print("\nDATA:\n")
                print(pre_transition_data)
                print("\n\n")
                raise e

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            reward, terminated, env_info = self.env.step(actions[0])

            # Here we added a modification for individually observed rewards
            total_reward = np.sum(np.array(reward))
            episode_return += total_reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1
            if photonallocated is not None:
                photonallocated.clear()
            photonallocated = actions.squeeze(dim=0).transpose(0,1).tolist()
            self.ps.storage_policy(selected_route, photonallocated, self.t)

        # print(f"Episode Return {episode_return}")

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        if not test_mode:
            self.t_env += self.t

        return self.batch, episode_return, CRR_times

    def get_env_info(self):
        return self.env.get_env_info()



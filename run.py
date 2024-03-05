import datetime
import os
import pprint
import time
import threading

import pandas as pd
import torch as th
import numpy as np

from types import SimpleNamespace as SN

from matplotlib import pyplot as plt

from ResourceAllocation.reward_decomposition import decompose_viz
from os.path import dirname, abspath
from torch.optim import RMSprop
from torch.optim import Adam

from ResourceAllocation.learners import REGISTRY as le_REGISTRY
from ResourceAllocation.runners import REGISTRY as r_REGISTRY
from ResourceAllocation.controllers import REGISTRY as mac_REGISTRY
from ResourceAllocation.components.episode_buffer import ReplayBuffer
from ResourceAllocation.components.transforms import OneHot

import ResourceAllocation.reward_decomposition.decompose as decompose
from ResourceAllocation.reward_decomposition.decomposer import RewardDecomposer
from TOQN import TOQNHyperparameters as tohp


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)

    # configure which device, attempting to use freest gpu available
    args.device = "cuda" if args.use_cuda else "cpu"

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token

    # Run and train
    run_sequential(args=args)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, delay_thr):
    args.env_args["learner_name"] = args.name

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args)

    # Set up schemes and groups here
    env_info = runner.get_env_info()

    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.obs_shape = env_info["obs_shape"]
    args.state_shape = env_info["state_shape"]
    args.graph_obj = env_info["graph_obj"]

    # Support Local Rewards also
    reward_shape = (1,)
    try:
        reward_shape = env_info["reward_shape"]
    except Exception as e:
        print("WARNING: Reward shape not specified in Enviroment, Assuming global reward")

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (env_info["request_num"],), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (tohp.request_num, env_info["n_actions"]), "group": "agents", "dtype": th.int},
        "reward": {"vshape": reward_shape},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions * tohp.request_num)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    args.reward_decomposer = RewardDecomposer(buffer.scheme, args) if args.decompose_reward else None
    # args.reward_optimiser = RMSprop(params=args.reward_decomposer.parameters(), lr=0.005, alpha=args.optim_alpha,
    #                                 eps=args.optim_eps) if args.decompose_reward else None
    args.reward_optimiser = Adam(params=args.reward_decomposer.parameters(),
                                 lr=0.01) if args.decompose_reward else None

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, args)

    if args.use_cuda:
        learner.cuda()

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1

    # draw_episode_reward()
    episode_rewards = []
    # draw_CRR_times
    CRR_times_s = []
    print("delay_thr is " + str(delay_thr))

    while runner.t_env <= args.t_max:
        if runner.t_env % 10000 == 0:
            print("t_env times is ----" + str(runner.t_env))
        runner.set_delay_thr(delay_thr)

        # Run for a whole episode at a time
        episode_batch, episode_return, CRR_times = runner.run(test_mode=False)
        episode_rewards.append(episode_return)
        CRR_times_s.append(CRR_times)
        buffer.insert_episode_batch(episode_batch)

        # First train the reward decomposer if necessary
        if args.decompose_reward and buffer.can_sample(args.reward_batch_size):
            for reward_update_idx in range(args.reward_updates_per_batch):
                reward_sample = buffer.sample(args.reward_batch_size)
                reward_sample.to(args.device)
                decompose.train_decomposer(args.reward_decomposer, reward_sample, args.reward_optimiser)

        # Save models to default directory
        # args.reward_decomposer.save_models()

        # Next Train the learner
        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            last_time = time.time()
            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        # if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
        #     model_save_time = runner.t_env
        #     save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
        #     # "results/models/{}".format(unique_token)
        #     os.makedirs(save_path, exist_ok=True)
        #
        #     # learner should handle saving/loading -- delegate actor save/load to mac,
        #     # use appropriate filenames to do critics, optimizer states
        #     learner.save_models(save_path)

        episode += args.batch_size_run

    draw_episode = {"x": [i for i in range(len(episode_rewards))], "episode_rewards": episode_rewards}
    pd.DataFrame(draw_episode).to_csv('./Draw/Question3/fig4/Node-cap-mean16.6.csv', index=False)
    # draw_CRR_times = {"x": [i for i in range(len(CRR_times_s))], "episode_rewards": CRR_times_s}
    # pd.DataFrame(draw_CRR_times).to_csv('./Draw/Question2/fig2/CRR_times_' + str(delay_thr) + '.csv', index=False)


def args_sanity_check(config):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        print("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config

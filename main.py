# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: Main                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import random
import numpy as np
import torch as th
from main_util import *
import run
from types import SimpleNamespace as SN
import threading

EPISODES = 2000

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return int(np.argmax(memory_available))

def run_name(env_name, alg_name, test_num, iteration_num, run_num):
    name = f"{test_num}-" if test_num is not None else ""
    name += f"{iteration_num}-" if iteration_num is not None else ""
    name += f"{run_num}-" if run_num is not None else ""
    name += f"{alg_name}-{env_name}"
    return name

def single_run(env_name, alg_name, seed, override_config=None, test_num=None, iteration_num=None, run_num=None):
    # Load algorithm and env base configs
    default_config = get_config_dict("default")
    env_config = get_config_dict(env_name, "envs")
    alg_config = get_config_dict(alg_name, "algs")
    if override_config is None:
        override_config = dict()

    # Load my Experiment Logger object for personal testing
    config_dict = default_config.copy()
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    config_dict = recursive_dict_update(config_dict, override_config)

    # Setting the random seed throughout the modules
    np.random.seed(seed)
    th.manual_seed(seed)
    random.seed(seed)
    config_dict["seed"] = seed

    config_dict = run.args_sanity_check(config_dict)

    # Set device for this experiment
    config_dict["device"] = "cpu"
    if config_dict["use_cuda"]:
        try:
            free_gpu_id = get_freer_gpu()
            config_dict["device"] = f"cuda:{free_gpu_id}"
            th.cuda.set_device(free_gpu_id)
        except Exception as e:
            print(f"Resorting to default cuda device, {e}")
            config_dict["device"] = "cuda"
    print(f"Running the test on device: {config_dict['device']}")

    # Setup logger and wandb. Modify current run name
    curr_run_name = run_name(env_name, alg_name, test_num, iteration_num, run_num)
    config_dict["run_name"] = curr_run_name

    # hyper parameters of constraints
    # delay_thr = 120
    # while delay_thr < 190:
    #     # Run the current test
    #     run.run_sequential(args=SN(**config_dict), delay_thr=delay_thr)
    #     delay_thr += 30
    run.run_sequential(args=SN(**config_dict), delay_thr=150)


    # Kill eveything
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

def main():
    # draw()
    # env_name = get_param(params, "--env-name")
    # alg_name = get_param(params, "--alg-name")
    env_name = "quantum_network"
    alg_name = "lomaq"

    # Seed is randomly generated for a single run like this
    seed = random.randrange(2 ** 32 - 1)

    single_run(env_name, alg_name, seed)


if __name__ == '__main__':
    main()

    # opr = OptimalRS()
    # ps = StoragePolicy()
    # env = QN()
    #
    # for episode in range(EPISODES):
    #     states, photonallocated = env.reset()
    #     step_counter = 0
    #     total_reward = [0 for i in range(tohp.nodes_num)]
    #     global_total_reward = 0
    #     while True:
    #         step_counter += 1
    #         # route selection
    #         opr.set_photon_allocation(photonallocated)
    #         # selected_route = opr.get_route_from_CRR(episode, ps)
    #         selected_route = [[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
    #         # ps.storage_policy(opr.get_Y(), photonallocated, episode)
    #         env.setSelectedRoutes(selected_route)
    #         states = env.transformStates(states)
    #         # resource allocation
    #         # actions = agents.choose_actionDQN(states)
    #         actions, log_probs = agents.choose_actionAC(states, episode, step_counter)
    #         next_states, rewards, global_reward, done = env.step(actions, step_counter)
    #         # agents.store_trans(states, actions, rewards, next_states)
    #         agents.store_trans(states, log_probs, rewards, next_states)
    #
    #         for i in range(tohp.nodes_num):
    #             total_reward[i] += rewards[i]
    #         global_total_reward += global_reward
    #
    #         if done:
    #             break
    #         if agents.memory_counter >= RLhp.MEMORY_CAPACITY:
    #             agents.learn()
    #         states = next_states
    #         photonallocated = agents.get_PApolicy(actions)
    #         # print("------Step_counter is " + str(step_counter))
    #     # print("In the" + str(episode) + " times transmission, the total throughput of reqSet R is " + str(total_reward))






import random
import pandas as pd
from ResourceAllocation.old.Agent import A2CAgent
from ResourceAllocation.old.QuantumNetwork import Env
import matplotlib.pyplot as plt
from Test.Example import HyperParameters as hp
from Test.Example.Route_Selection import RouteSelectionILP


def decentralized (episode, selected_routes, env, net):
    step_counter = 0
    total_reward = 0
    observations = env.reset(step_counter, selected_routes)
    next_all_actions = []
    while True:
        actions = net.choose_action(observations)
        next_all_actions.append(actions)
        next_observations, reward, done = env.step(step_counter, actions)
        net.store_trans(observations, actions, reward, next_observations)

        total_reward += reward
        if net.memory_counter >= net.memory_capacity:
            net.learn()

        step_counter += 1
        if done or step_counter>=hp.STEP_LIMITATION:
            break
        observations = next_observations

    print("episode {}, the reward is {}".format(episode, round(total_reward / step_counter, 3)))
    return total_reward/step_counter, next_all_actions


def centralized_assistant(step_counter, actions):
    assistant = RouteSelectionILP(step_counter, actions)
    return assistant.integerRun()


if __name__ == '__main__':
    # 3*18
    actions = [[random.randint(0, hp.N_ACTIONS) for a in range(hp.AGENT_NUM)] for r in range(hp.REQUEST_NUM)]
    selected_routes, min_delay = centralized_assistant(-1, actions)

    env = Env(selected_routes)
    net = A2CAgent()
    acc_reward = []
    axx = []
    acc_delay = []
    for epi in range(hp.EPISODES):
        print(str(epi) + '---------------------')
        total_reward, next_epi_actions = decentralized(epi, selected_routes, env, net)
        acc_reward.append(total_reward)
        axx.append(epi)
        net.plot(net.ax, acc_reward)

        actions.clear()
        actions = next_epi_actions
        next_selected_routes, min_delay = centralized_assistant(epi, actions)
        selected_routes.clear()
        selected_routes = next_selected_routes
        acc_delay.append(min_delay)

    fig, ax = plt.subplots()
    ax.plot(axx, acc_reward, color='green')
    ax.set_ylabel('Total_throughput')
    ax.set_xlabel('Episodes')
    ax.spines['right'].set_visible(False)

    z_ax = ax.twinx()
    z_ax.plot(axx, acc_delay, color='blue')
    z_ax.set_ylabel('Total_delay')

    plt.show()

    # plt.xlabel("episodes")
    # plt.ylabel("total fidelity")
    # plt.plot(axx, acc_reward, 'b-')
    # plt.show()
    res = {"x": axx, "acc_throughput": acc_reward}
    pd.DataFrame(res).to_csv('./results/Total-delay-reward-lr0.001-4.csv', index=False)
    res2 = {"x": axx, "acc_delay": acc_delay}
    pd.DataFrame(res2).to_csv('./results/Total-delay-centralized-ILP-4.csv', index=False)











import random

import pandas as pd
from Agent import A2CAgent
from QuantumNetwork import Env
import matplotlib.pyplot as plt
import HyperParameters as hp
from Route_Selection import RouteSelectionILP


def decentralized (selected_routes):
    env = Env(selected_routes)
    net = A2CAgent()
    print("The A2C agents are collecting experience...")
    acc_reward = []
    axx = []
    for episode in range(hp.EPISODES):
        step_counter = 0
        total_reward = 0
        observations = env.reset()

        while True:
            step_counter += 1
            actions = net.choose_action(observations)
            selected_routes = centralized_assistant(step_counter, actions)
            next_observations, reward, done = env.step(actions, step_counter)
            net.store_trans(observations, actions, reward, next_observations)

            total_reward += reward
            if net.memory_counter >= net.memory_capacity:
                net.learn()

            if done:
                break
            observations = next_observations

        print("episode {}, the reward is {}".format(episode, round(total_reward / step_counter, 3)))
        acc_reward.append(total_reward / step_counter)  # total_reward/step_counter
        axx.append(episode)
        net.plot(net.ax, acc_reward)

    plt.xlabel("episodes")
    plt.ylabel("total fidelity")
    plt.plot(axx, acc_reward, 'b-')
    plt.show()
    res = {"x": axx, "acc_reward": acc_reward}
    pd.DataFrame(res).to_csv('Total-fidelity-reward-lr0.001.csv', index=False)

def centralized_assistant(step_counter, actions):
    assistant = RouteSelectionILP(step_counter, actions)
    return assistant.integerRun()


if __name__ == '__main__':
    # initialize
    # initialize
    selected_routes = [random.randint(0, hp.CANDIDATE_ROUTE_NUM-1) for i in range(hp.REQUEST_NUM)]
    decentralized(selected_routes)


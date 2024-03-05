import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def take_mean_value(y):
    y_new = []
    for i in range(20):
        y_new.append(sum(y[0:20])/20)
    for t in range(0, 80):
        y_new.append(sum(y[t:t+20]) / 20)
    for t in range(0, len(y)-100):
        y_new.append(sum(y[t:t+100])/100)
    return y_new

OTOH = pd.read_csv("OTOH.csv")
x = [i for i in range(len(OTOH['x']))]
OTOH_episode_reward = OTOH['episode_rewards']
avg_OTOH_episode_reward = take_mean_value(OTOH['episode_rewards'])

fig = plt.figure()
plt.plot(x, OTOH_episode_reward, alpha=0.5, color='r', label='')
plt.plot(x, avg_OTOH_episode_reward, color='r', marker='s', markevery=1000, label='OTOH')
plt.gca().yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText = True))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.tick_params(labelsize=12)
plt.legend(fontsize=13)
plt.xlabel('Episodes', fontsize=15)
plt.ylabel("Episode reward", fontsize=15)
plt.grid()
plt.show()
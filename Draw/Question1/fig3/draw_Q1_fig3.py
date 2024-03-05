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

OTOH = pd.read_csv("Agent_rewards_2W.csv")
x = [i for i in range(len(OTOH['x']))]
avg_OTOH_episode_reward = take_mean_value(OTOH['4'])
# avg_CRR_episode_reward = take_mean_value(OTOH['10'])
avg_CRR_episode_reward2 = take_mean_value(OTOH['17'])

fig = plt.figure()
plt.plot(x, avg_OTOH_episode_reward, color='#344F99', marker='s', markevery=1000, label='Agent 5')
# plt.plot(x, avg_CRR_episode_reward, color='b', marker='o', markevery=1000, label='OTOH-noCRR')
plt.plot(x, avg_CRR_episode_reward2, color='#A14D61', marker='o', markevery=1000, label='Agent 18')
plt.gca().yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText = True))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.tick_params(labelsize=12)
plt.legend(fontsize=13)
plt.xlabel('Episodes', fontsize=15)
plt.ylabel("Agent episode reward", fontsize=15)
plt.grid()
plt.show()
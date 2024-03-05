import pandas as pd
import matplotlib.pyplot as plt


def take_mean_value(y):
    y_new = []
    for i in range(20):
        y_new.append(sum(y[0:20])/20)
    for t in range(0, 80):
        y_new.append(sum(y[t:t+20]) / 20)
    for t in range(0, len(y)-100):
        y_new.append(sum(y[t:t+100])/100)
    return y_new

data = pd.read_csv("CRR_times_120.csv")
data2 = pd.read_csv("CRR_times_150.csv")
data3 = pd.read_csv("CRR_times_180.csv")
x = [i for i in range(len(data['x']))]
step_avg_data = [d for d in data['episode_rewards']]
episode_reward = take_mean_value(step_avg_data)
step_avg_data2 = [d for d in data2['episode_rewards']]
episode_reward2 = take_mean_value(step_avg_data2)
step_avg_data3 = [d for d in data3['episode_rewards']]
episode_reward3 = take_mean_value(step_avg_data3)

fig = plt.figure()
plt.plot(x, episode_reward, marker='o', markevery=150, color='m', linestyle='-', markerfacecolor='none', label='$\Gamma^{thr}=120$')
plt.plot(x, episode_reward2, marker='^', markevery=150, color='c', linestyle='-.', markerfacecolor='none', label='$\Gamma^{thr}=150$')
plt.plot(x, episode_reward3, marker='s', markevery=150, color='y', linestyle='dashed', markerfacecolor='none', label='$\Gamma^{thr}=180$')
plt.tick_params(labelsize=12)
plt.legend(fontsize=13)
plt.xlabel('Episodes', fontsize=15)
plt.ylabel("Average CRR used times", fontsize=15)
plt.grid()
plt.show()
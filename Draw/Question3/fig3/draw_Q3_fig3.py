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

data = pd.read_csv("request-4.csv")
data2 = pd.read_csv("request-5.csv")
data3 = pd.read_csv("request-6.csv")
data4 = pd.read_csv("request-8.csv")
x = [i for i in range(len(data['x']))]
step_avg_data = [d/4 for d in data['episode_rewards']]
episode_reward = take_mean_value(step_avg_data)
step_avg_data2 = [d/5 for d in data2['episode_rewards']]
episode_reward2 = take_mean_value(step_avg_data2)
step_avg_data3 = [d/6 for d in data3['episode_rewards']]
episode_reward3 = take_mean_value(step_avg_data3)
step_avg_data4 = [d/8 for d in data4['episode_rewards']]
episode_reward4 = take_mean_value(step_avg_data4)

fig = plt.figure()
plt.plot(x, episode_reward, marker='o', markevery=150, color='#F9E2AF', linestyle='-', markerfacecolor='none', label='$r\_bs$=4')
plt.plot(x, episode_reward2, marker='^', markevery=150, color='#009FBD', linestyle='-.', markerfacecolor='none', label='$r\_bs$=5')
plt.plot(x, episode_reward3, marker='s', markevery=150, color='#210062', linestyle='dashed', markerfacecolor='none', label='$r\_bs$=6')
plt.plot(x, episode_reward4, marker='*', markevery=150, color='#77037B', linestyle='dotted', markerfacecolor='none', label='$r\_bs$=8')
plt.tick_params(labelsize=12)
plt.legend(fontsize=13)
plt.xlabel('Episodes', fontsize=15)
plt.ylabel("Average episode reward/r_bs", fontsize=15)
plt.grid()
plt.show()
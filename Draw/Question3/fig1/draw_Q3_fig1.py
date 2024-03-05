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

data = pd.read_csv("LR-0.01.csv")
data2 = pd.read_csv("LR-0.05.csv")
data3 = pd.read_csv("LR-0.005.csv")
x = [i for i in range(len(data['x']))]
episode_reward = take_mean_value(data['episode_rewards'])
episode_reward2 = take_mean_value(data2['episode_rewards'])
episode_reward3 = take_mean_value(data3['episode_rewards'])

fig = plt.figure()
plt.plot(x, episode_reward, marker='o', markevery=150, color='#F9E2AF', linestyle='-', markerfacecolor='none', label='LR=0.01')
plt.plot(x, episode_reward2, marker='^', markevery=150, color='#009FBD', linestyle='-.', markerfacecolor='none', label='LR=0.05')
plt.plot(x, episode_reward3, marker='s', markevery=150, color='#77037B', linestyle='dashed', markerfacecolor='none', label='LR=0.005')
plt.tick_params(labelsize=12)
plt.legend(fontsize=13)
plt.xlabel('Episodes', fontsize=15)
plt.ylabel("Average episode reward", fontsize=15)
plt.grid()
plt.show()
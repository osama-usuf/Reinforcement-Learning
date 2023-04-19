import numpy as np 
import matplotlib.pyplot as plt

runs = 10

accumulated_rewards_all = []
episode_rewards_all = []
for run in range(runs):
    episode_rewards = np.loadtxt(f'data/SARSA_{run}.txt')
    accumulated_rewards = []
    for i in range(len(episode_rewards)):
        accumulated_reward = np.average(episode_rewards[:i])
        accumulated_rewards.append(accumulated_reward)
    accumulated_rewards_all.append(accumulated_rewards)
    episode_rewards_all.append(episode_rewards)

avg_accumulated_reward = np.average(np.array(accumulated_rewards_all), axis=0)
plt.xlabel('asd')
plt.plot(avg_accumulated_reward)
plt.show()
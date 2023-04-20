import numpy as np 
import matplotlib.pyplot as plt



def get_avg_accumulated_reward(algorithm, runs):
    accumulated_rewards_all = []
    episode_rewards_all = []
    for run in range(runs):
        episode_rewards = np.loadtxt(f'data/{algorithm}_{run}.txt')
        accumulated_rewards = []
        for i in range(1, len(episode_rewards)):
            accumulated_reward = np.average(episode_rewards[:i])
            accumulated_rewards.append(accumulated_reward)
        accumulated_rewards_all.append(accumulated_rewards)
        episode_rewards_all.append(episode_rewards)

    avg_accumulated_reward = np.average(np.array(accumulated_rewards_all), axis=0)
    return avg_accumulated_reward

runs = 10
sarsa_rewards = get_avg_accumulated_reward('SARSA', runs=runs)
qlearning_rewards = get_avg_accumulated_reward('QLearning', runs=runs)


with plt.style.context('fivethirtyeight'):
    figsize = (8, 8)
    plt.figure(figsize=figsize)
    plt.xlabel('Episode (#)')
    plt.ylabel('Average accumulated reward')
    plt.tight_layout()
    plt.ylim(-1.5, 3)
    plt.plot(sarsa_rewards)
    plt.savefig('plots/sarsa.png', dpi='figure')

    plt.figure(figsize=figsize)
    plt.xlabel('Episode (#)')
    plt.ylabel('Average accumulated reward')
    plt.tight_layout()
    plt.plot(qlearning_rewards, c='red')
    plt.ylim(-1.5, 3)
    plt.savefig('plots/qlearning.png', dpi='figure')

    plt.figure(figsize=figsize)
    plt.xlabel('Episode (#)')
    plt.ylabel('Average accumulated reward')
    plt.plot(sarsa_rewards, label='SARSA')
    plt.plot(qlearning_rewards, label='QLearning', c='red')
    plt.tight_layout()
    plt.legend(fontsize=24)
    plt.ylim(-1.5, 3)
    plt.savefig('plots/comparison.png', dpi='figure')
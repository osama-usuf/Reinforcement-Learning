import matplotlib.pyplot as plt
import numpy as np

def plot_average_reward_actions(bandit):

    mean_rewards = bandit.get_average_statistics()

    # Plots can be modified here
    x = list(range(len(mean_rewards)))
    assert x == list(range(len(mean_actions)))
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle(f'Multi-arm Bandit Solver with {len(bandit.arms)} arm(s)\n{bandit.policy_name} Policy')
    axs[0].plot(x, mean_rewards)
    axs[1].scatter(x, mean_actions, color='orange', alpha=0.3)
    axs[0].set(ylabel='Reward (r)')
    axs[1].set(ylabel='Action (a)')
    axs[1].set_yticks(list(range(len(bandit.arms))))
    plt.xlabel('Timestep (t)')

def plot_average_reward(bandit, label, title):
    mean_rewards = bandit.get_average_rewards()
    mean_rewards = [np.average(mean_rewards[:i]) for i in range(1, len(mean_rewards))]
    # Plots can be modified here
    x = list(range(len(mean_rewards)))
    plt.title(f'Multi-arm Bandit Solver with {len(bandit.arms)} arm(s)\n{bandit.policy_name.capitalize()} Policy with {title}')
    plt.plot(x, mean_rewards, label=label)
    plt.ylabel(f'Accumulated Reward (r)\nAveraged over {bandit.runs} independent runs')
    #plt.ylim(-0.5, 8)
    plt.xlabel('Timestep (t)')

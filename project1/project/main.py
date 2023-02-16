"""
    ECE 6882: Reinforcement Learning @ The George Washington University, Spring 2023.
    Project 1: Multi-arm Bandit Problem

    Environment Description:
    Think of an agent that plays a 2-armed bandit, trying to maximize its total reward. In each step,
    the agent selects one of the levers and is given some reward according to the reward distribution
    of that lever. Assume that reward distribution for the first lever is a Gaussian with 𝜇 = 5, 𝜎² = 10, 
    and for the second lever is a binomial Gaussian with 𝜇_1 = 10, 𝜎_2² = 15, and 𝜇_1 = 4, 𝜎_2² = 10,
    which means that the resulting output will be uniformly probable from these two Gaussian
    distributions (See http://en.wikipedia.org/wiki/Mixture_distribution).
    Implement this environment in Python together with a random policy that chooses the two
    actions with equal probabilities. Plot the resulting average reward per timestep obtained by the
    agent up to timestep 𝑘 for 𝑘 = 1, …, 1000.

    Coded by: Osama Yousuf (osamayousuf@gwu.edu)
"""

import argparse

from util import Bandit, Arm, RandomPolicy, LearningRates
from plots import *


def simulate_bandit(args):
    """
        Main function for simulating the multi-arm bandit problem. Uses helper classes defined in the complementary util.py file.
    """

    # Initialize an array of Arm objects. Each Arm object holds underlying distribution information.
    # Refer to the Arm class definition for information on what arguments need to be passed.
    arms = [
            Arm(distribution_type='gaussian', distribution_args=(5, 10)),
            Arm(distribution_type='mixed gaussian', distribution_args=[(10, 15), (4, 10)])
           ]

    # A constant distribution can be added as follows:
    # arms.append(Arm(distribution_type='constant', distribution_args=(5)))
    # random_policy = ('random', {}) # (policy_name, policy_args)

    exp_a, exp_b, exp_c = True, True, True

    # Part (a)
    # Epsilon-Greedy-Policy
    # Initial Q values = [0, 0]
    # Epsilon = [0, 0.1, 0.2, 0.5]
    # Alphas = 1, 0.9^k, 1/1+Ln(1+k), 1/k

    if (exp_a):
        print('Experiment a')

        alphas_all = [(LearningRates.constant, 'α = 1'),
                    (LearningRates.exponential, 'α = 0.9ᵏ'),
                    (LearningRates.logarithmic, 'α = 1 / (1 + ln(1+k))'),
                    (LearningRates.inverse, 'α = 1 / k')]
        initial_q_values_all = [[0, 0]]
        epsilons_all = [0, 0.1, 0.2, 0.5]
        initial_q_values = initial_q_values_all[0]

        fig_num = 1
        for alpha_idx in range(len(alphas_all)):
            plt.figure()
            for epsilon_idx in range(len(epsilons_all)):
                # TODO - CHECK IF Q VALUES INITIAL ARE ALWAYS 0
                alpha, title = alphas_all[alpha_idx]
                epsilon = epsilons_all[epsilon_idx]
                label = f'ε = {epsilon}'
                greedy_policy = ('greedy', {'alpha': alpha, 
                                            'initial_q_values':initial_q_values.copy(),
                                            'epsilon': epsilon})

                bandit = Bandit(arms=arms, policy=greedy_policy, timesteps=args.k, runs=args.r)
                bandit.simulate()
                plot_average_reward(bandit, label=label, title=title)
                # Print statistics for the Tables
                print(title, label, bandit.get_average_qs())
            plt.legend()
            plt.savefig(f'a-Figure{fig_num}.png', dpi='figure')
            fig_num += 1

    # Part (b)
    # Epsilon-Greedy-Policy
    # Initial Q values = [[0, 0], [5, 7], [20, 20]]
    # Epsilon = 0.1
    # Alphas = 0.1

    if (exp_b):
        print('Experiment b')
        epsilon = 0.1
        alpha = LearningRates.constant_fractional
        title = f'α = 0.1, ε = {epsilon}'
        initial_q_values_all = [[0, 0], [5, 7], [20, 20]]
        
        fig_num = 1
        
        plt.figure()
        for initial_q_values_idx in range(len(initial_q_values_all)):
            greedy_policy = ('greedy', {'alpha': alpha, 
                                        'initial_q_values':initial_q_values_all[initial_q_values_idx].copy(),
                                        'epsilon': epsilon})

            bandit = Bandit(arms=arms, policy=greedy_policy, timesteps=args.k, runs=args.r)
            bandit.simulate()
            label = f"Q = {initial_q_values_all[initial_q_values_idx]}"
            plot_average_reward(bandit, label=label, title=title)
            # Print statistics for the Tables
            print(title, label, bandit.get_average_qs())
        plt.legend()
        plt.savefig(f'b-Figure{fig_num}.png', dpi='figure')

    # Part (c)
    # Gradient-Based Policy
    # Alpha = 0.1
    # Preferences = 0

    if (exp_c):
        print('Experiment c')
        alpha = LearningRates.constant_fractional
        H = [0, 0]
        title = f'α = 0.1, H = {H}'
        fig_num = 1
        
        plt.figure()
        gradient_policy = ('gradient', {'alpha': alpha, 'preferences':H})

        bandit = Bandit(arms=arms, policy=gradient_policy, timesteps=args.k, runs=args.r)
        bandit.simulate()
        label = title
        plot_average_reward(bandit, label=label, title=title)
        # Print statistics for the Tables
        print(title, 'Final Hs', bandit.get_average_hs(), 'Final probs', bandit.get_average_pis(), 'Final R_bar', np.average(bandit.get_average_rewards()))
        plt.legend()
        plt.savefig(f'c-Figure{fig_num}.png', dpi='figure')


if __name__ == "__main__":
    np.random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=1000, type=int, help="Timesteps/Iterations to simulate")
    parser.add_argument("--r", default=200, type=int, help="Runs/Times to execute the k iterations")
    args = parser.parse_args()
    simulate_bandit(args)
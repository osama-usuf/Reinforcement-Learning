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

from util import Bandit, Arm

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

    # Initialize the Bandit object, simulate and plot information.
    # By default, the implemented policy is random. 
    bandit = Bandit(arms=arms)
    bandit.simulate(timesteps=args.k)
    bandit.plot_info()

    print(f"Average reward: {bandit.get_average_reward()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=1000, type=int, help="Timesteps/Iterations to simulate")
    args = parser.parse_args()
    print(f"Number of timesteps: {args.k}")
    simulate_bandit(args)
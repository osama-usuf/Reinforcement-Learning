import matplotlib.pyplot as plt
import numpy as np

class Arm:
    def __init__(self, distribution_type, distribution_args):
        '''
            The Arm object stores information about the underlying distribution. 

            distribution_type: type of distribution. Valid choices: ['constant', 'gaussian', 'mixed gaussian']
            distribution_args: corresponding arguments for a given distribution_type.
                distribution_type | distribution_args
                -------------------------------------
                constant          | a single float element c i.e. (c)
                gaussian          | tuple of two floats, one for mean µ, another for variance σ i.e. (µ, σ²). σ can not be negative
                mixed_gaussian    | a list of n tuples, where n > 1 (2, 3, ...). Each tuple defines a gaussian distribution i.e. (µ, σ²), where σ can not be negative
        '''
        if (distribution_type) not in  ['constant', 'gaussian', 'mixed gaussian']:
            print('Arm distribution not defined.')
            exit()
        self.distribution_type = distribution_type
        self.distribution_args = distribution_args
        
    def pull(self):
        """
            Returns a single sample from the underlying distribution
        """
        # Sample from the underlying distribution
        # Additional distribution definitions would go here
        if (self.distribution_type == 'constant'):
            constant = self.distribution_args
            return constant
        elif (self.distribution_type == 'gaussian'):
            mean, variance = self.distribution_args
            standard_deviation = np.sqrt(variance)
            sample = np.random.normal(mean, standard_deviation, 1)[0]
            return sample
        elif (self.distribution_type == 'mixed gaussian'):
            num_gaussians = len(self.distribution_args)
            # For each gaussian in the mixture, we need exactly two arguments (µ, σ²). 
            # The following assertion checks for validity (more than 1 gaussian is present)
            assert num_gaussians > 1
            # We gather a single sample from each distribution in the mixture. The `samples` list simply stores these
            samples = []
            for mean, variance in self.distribution_args:
                standard_deviation = np.sqrt(variance)
                sample = np.random.normal(mean, standard_deviation, 1)[0]
                samples.append(sample)
            # The final returned sample is simply one of the samples present in the `samples` list.
            # This means that the resulting output will be uniformly probable from all the distributions present in the mixture.
            # To assign weights to each distribution, an additional probability array can be provided. Refer to: https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
            final_sample = np.random.choice(samples)
            return final_sample
        else:
            print('Arm distribution not defined.')
            exit()

class Bandit:
    """
        The Bandit object holds information about the agent as well as the multi-arm bandit environment
    """
    def __init__(self, arms):
        self.arms = arms
        assert len(self.arms) > 0
        self.rewards = []
        self.actions = []
        # Additional attributes/metrics can be added here
        
    def simulate(self, timesteps):
        for i in range(timesteps):
            # Random policy - randomly choose an arm to pull between the available arms. 
            # variable 'action' should store the index of the arm to be pulled
            # Custom policies would go here (comment out/remove random policy)
            action = np.random.randint(0, len(self.arms)) 

            # To take the action based on a policy, simply call the pull() method of the arm at corresponding index 
            reward = self.arms[action].pull()
            # Append necessary information to the bandit object, makes plotting stuff later easier
            self.rewards.append(reward)
            self.actions.append(action)

    def plot_info(self):
        # Plots can be modified here
        x = list(range(len(self.rewards)))
        assert x == list(range(len(self.actions)))
        fig, axs = plt.subplots(2, sharex=True)
        fig.suptitle(f'Multi-arm Bandit Solver with {len(self.arms)} arm(s)\nRandom Policy')
        axs[0].plot(x, self.rewards)
        axs[1].scatter(x, self.actions, color='orange', alpha=0.3)
        axs[0].set(ylabel='Reward (r)')
        axs[1].set(ylabel='Action (a)')
        axs[1].set_yticks(list(range(len(self.arms))))
        plt.xlabel('Timestep (t)')
        plt.show()

    def get_average_reward(self):
        if (len(self.rewards) == 0): return 0
        else: return sum(self.rewards) / len(self.rewards) 
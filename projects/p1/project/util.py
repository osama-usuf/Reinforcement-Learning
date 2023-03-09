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
    def __init__(self, arms, policy, timesteps, runs):
        self.arms = arms
        assert len(self.arms) > 0
        self.actions = []

        policy_name, policy_args = policy
        if policy_name == 'random': self.policy = RandomPolicy(arms)
        elif policy_name == 'greedy': self.policy = GreedyPolicy(arms, **policy_args)
        elif policy_name == 'gradient': self.policy = GradientPolicy(arms, **policy_args)
        else: 
            print('Undefined policy specified.')
            exit()
        self.policy_name = policy_name

        self.timesteps = timesteps
        self.runs = runs
        
        self.rewards = np.zeros((self.runs, self.timesteps+1))
        self.actions = np.zeros((self.runs, self.timesteps+1))

        # Greedy policy
        self.final_qs_from_policy = np.zeros((self.runs, len(self.arms)))
        # Gradient policy
        self.final_hs_from_policy = np.zeros((self.runs, len(self.arms)))
        self.final_pis_from_policy = np.zeros((self.runs, len(self.arms)))
        
    def simulate(self, verbose=False):
        for j in range(self.runs):
            if (verbose): print(f'Run {j}', self.policy.H, self.policy.pis)
            for i in range(1, self.timesteps+1):
                # Random policy - randomly choose an arm to pull between the available arms. 
                # variable 'action' should store the index of the arm to be pulled
                # Custom policies would go here (comment out/remove random policy)
                
                action = self.policy.get_action()
                # To take the action based on a policy, simply call the pull() method of the arm at corresponding index 
                reward = self.arms[action].pull()
                # Append necessary information to the bandit object, makes plotting stuff later easier
                self.rewards[j, i] = reward
                self.actions[j, i] = action

                # Call the policy's update functions as needed 
                if self.policy_name == 'greedy': self.policy.update_q_values(action, reward, i)
                elif self.policy_name == 'gradient': self.policy.update_preferences(action, reward, i)
            if self.policy_name == 'greedy': self.final_qs_from_policy[j] = self.policy.Q
            elif self.policy_name == 'gradient': 
                self.final_hs_from_policy[j] = self.policy.H
                self.final_pis_from_policy[j] = self.policy.pis
            # Run complete, reset policy parameters
            if (verbose): print(f'Run {j}', self.policy.H, self.policy.pis)
            self.policy.reset()

    def get_average_rewards(self):
        return np.mean(self.rewards, axis=0)

    def get_average_qs(self):
        return np.mean(self.final_qs_from_policy, axis=0)

    def get_average_hs(self):
        return np.mean(self.final_hs_from_policy, axis=0)

    def get_average_pis(self):
        return np.mean(self.final_pis_from_policy, axis=0)

# Learning Rate functions
class LearningRates:
    @staticmethod
    def constant(k):
        return 1
    def constant_fractional(k):
        return 0.1
    @staticmethod
    def exponential(k):
        return 0.9**k
    @staticmethod
    def logarithmic(k):
        return 1 / (1 + np.log(1+k))
    @staticmethod
    def inverse(k):
        return 1 / k
    

# Policies
class RandomPolicy:
    """
        Randomly pick an action
    """
    def __init__(self, arms):
        self.arms = arms

    def get_action(self):
        action = np.random.randint(0, len(self.arms))
        return action
    def reset(self):
        pass


class GreedyPolicy:
    """
        Epsilon-greedy policy
    """
    def __init__(self, arms, alpha, initial_q_values, epsilon):
        self.alpha = alpha
        self.arms = arms
        self.Q = initial_q_values
        self.initial_q_constant = initial_q_values
        self.epsilon = epsilon
        assert epsilon >= 0 and epsilon <= 1

    def get_action(self):
        # generate a random uniform number
        toss = np.random.random()
        if toss <= 1 - self.epsilon: # greedy choice, a = argmax(Q_i)
            action = np.argmax(self.Q)
        else:
            action = np.random.randint(0, len(self.arms)) # random choice
        return action

    def update_q_values(self, action, reward, timestep):
        self.Q[action] = self.Q[action] + self.alpha(timestep) * (reward - self.Q[action])

    def reset(self):
        self.Q = self.initial_q_constant

class GradientPolicy:
    """
        Gradient-Bandit policy
    """
    def __init__(self, arms, alpha, preferences):
        self.alpha = alpha
        self.arms = arms
        self.H = np.array(preferences, dtype=np.float32)
        self.initial_H_constant = np.array(preferences, dtype=np.float32)
        assert len(preferences) == len(arms)
        self.rewards = [0] # list of rewards, added every time step, useful in calculating R_bar = r1 + r2 + ... + rt / t
        self.pis = np.exp(self.H) / np.sum(np.exp(self.H))

    def get_average_rewards(self):
        return np.mean(self.rewards)

    def get_action(self):
        action = np.random.choice(list(range(len(self.arms))), size=1, replace=False, p=self.pis)[0]
        return action

    def update_preferences(self, action, reward, timestep):
        self.rewards.append(reward)
        r_bar = self.get_average_rewards()
        # TODO: can technically be vectorized using numpy
        for arm in range(len(self.arms)):
            if arm == action: # action selected
                #print('updating', self.H)
                self.H[arm] += self.alpha(timestep) * (reward - r_bar) * (1 - self.pis[arm])
                #print('updated', self.H)
            else:
                self.H[arm] -= self.alpha(timestep) * (reward - r_bar) * (self.pis[arm])
            
        self.pis = np.exp(self.H) / np.sum(np.exp(self.H))
        
    def reset(self):
        self.rewards = [0]
        self.H = np.copy(self.initial_H_constant)       

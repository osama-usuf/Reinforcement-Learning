import numpy as np

class RandomPolicy:
    def __init__(self, maze):
        self.maze = maze
        self.initialize_policy_attrs()

    def initialize_policy_attrs(self):
        # Initialize policy parameters
        # Action Space
        # Up = 0, Down = 1, Left = 2, Right = 3
        self.pi = np.random.choice(self.maze.actions, size=self.maze.data.shape)

    def step_rand_dir(self):
        # Pick an action randomly
        return np.random.choice(self.maze.actions, replace=False)

    def learn_policy(self):
        # Does not account for transition probabilities
        return self.pi


class PolicyIteration:
    def __init__(self, maze, gamma, theta):
        '''
            gamma = discount factor in the reward; discounts future reward and guarantees convergence
            theta = a small +ve no. that determines the accuracy of estimation of the policy during evaluation
        '''
        self.maze = maze
        self.gamma, self.theta = gamma, theta
        self.initialize_policy_attrs()
        self.max_value_iters = 1000
        self.max_policy_iters = 1000

    def initialize_policy_attrs(self):
        # Initialize policy parameters
        self.V = np.zeros_like(self.maze.data)
        # Action Space
        # Up = 0, Down = 1, Left = 2, Right = 3
        self.pi = np.ones_like(self.maze.data) * 2

    def learn_policy(self):
        for i in range(self.max_policy_iters):
            # Policy Evaluation
            iter_num = 0 
            while iter_num < self.max_value_iters:
                delta = 0
                for iy, ix in np.ndindex(self.V.shape): # Loop over states
                    # print(self.V[iy, ix])
                    v = self.V[iy, ix]
                    # Update V
                    summation = 0
                    next_states = self.maze.get_next_states(iy, ix, self.pi[iy, ix])
                    for s, r, p in next_states: summation += p * (r + self.gamma * self.V[s[0], s[1]])
                    self.V[iy, ix] = summation
                    delta = max(delta, abs(v - self.V[iy, ix]))

                if (delta < self.theta): 
                    print(f"Policy Iteration converged in {iter_num}/{self.max_value_iters} iterations.")
                    break
                iter_num += 1

            # Policy Improvement
            policy_stable = True
            for iy, ix in np.ndindex(self.pi.shape): 
                old_action = self.pi[iy, ix]
                action_outcomes = []
                # update the action if a) it improves value, b) it is different from current policy
                for action in range(len(self.maze.action_space)):
                    summation = 0
                    next_states = self.maze.get_next_states(iy, ix, action)
                    for s, r, p in next_states: summation += p * (r + self.gamma * self.V[s[0], s[1]])
                    action_outcomes.append(summation)

                self.pi[iy, ix] = np.argmax(action_outcomes)
                if (old_action != self.pi[iy, ix]):
                    policy_stable = False

            if policy_stable:
                print(f'Stable Policy found in {i}/{self.max_policy_iters} iterations.')
                return self.pi # return optimal stable policy
        # for completeness, we do return unstable policy if not found
        return self.pi


class ValueIteration:
    def __init__(self, maze, gamma, theta):
        '''
            transition_randomness = p. Agent moves to indended state with 1 - p probability, and a random neighbor with p / 3 probability.
            gamma = discount factor in the reward; discounts future reward and guarantees convergence
            theta = a small +ve no. that determines the accuracy of estimation of the policy during evaluation
        '''
        self.maze = maze
        self.gamma, self.theta = gamma, theta
        self.initialize_policy_attrs()
        self.max_value_iters = 1000
        self.max_policy_iters = 1000

    def initialize_policy_attrs(self):
        # Initialize policy parameters
        self.V = np.zeros_like(self.maze.data)
        # Action Space
        # Up = 0, Down = 1, Left = 2, Right = 3
        self.pi = np.ones_like(self.maze.data) * 2

    def learn_policy(self):
        # for i in range(self.max_policy_iters):
        #     # Policy Evaluation
        iter_num = 0 
        while iter_num < self.max_value_iters:
            delta = 0
            for iy, ix in np.ndindex(self.V.shape): # Loop over states
                v = self.V[iy, ix]
                # Update V
                action_outcomes = []
                for action in range(len(self.maze.action_space)):
                    summation = 0
                    next_states = self.maze.get_next_states(iy, ix, action)
                    for s, r, p in next_states: summation += p * (r + self.gamma * self.V[s[0], s[1]])
                    action_outcomes.append(summation)
                self.V[iy, ix] = np.max(action_outcomes)
                delta = max(delta, abs(v - self.V[iy, ix]))

            if (delta < self.theta): 
                print(f"Value Iteration converged in {iter_num}/{self.max_value_iters} iterations.")
                break
            iter_num += 1

        # Output a deterministic policy
        for iy, ix in np.ndindex(self.pi.shape): 
            action_outcomes = []
            for action in range(len(self.maze.action_space)):
                summation = 0
                next_states = self.maze.get_next_states(iy, ix, action)
                for s, r, p in next_states: summation += p * (r + self.gamma * self.V[s[0], s[1]])
                action_outcomes.append(summation)
            self.pi[iy, ix] = np.argmax(action_outcomes)

        return self.pi

class SARSA:
    def __init__(self, maze, gamma, alpha, epsilon):
        '''
            gamma = discount factor in the reward; discounts future reward and guarantees convergence
            alpha = TD learning rate. The factor which the TD error gets multiplied with
            epsilon = epsilon for the epsilon-greedy step
        '''
        self.maze = maze
        self.gamma, self.alpha, self.epsilon = gamma, alpha, epsilon
        self.episodes = 1000
        self.steps = 1000
        self.initialize_policy_attrs()

    def initialize_policy_attrs(self):
        # Initialize policy parameters
        self.n_actions = len(self.maze.actions)
        # Initialize Q matrix for all (state, action) pairs
        self.Q = np.zeros((*self.maze.data.shape, self.n_actions))
        # Action Space
        # Up = 0, Down = 1, Left = 2, Right = 3
        self.pi = np.ones_like(self.maze.data) * 2
        # self.rewards = np.zeros((self.episodes, self.steps))
        self.episode_rewards = []

    def greedy_action(self, s):
        # draw an epsilon greedy action
        # generate a random uniform number
        toss = np.random.random()
        if toss <= 1 - self.epsilon: # greedy choice, a = argmax(Q_i)
            action = np.argmax(self.Q[s[0], s[1]])
        else:
            action = np.random.randint(0, self.n_actions) # random choice
        return action

    def learn_policy(self):
        for episode in range(self.episodes):
            episode_reward = 0
            # Print every 100 episodes
            if (episode % 100 == 0): print(f'Episode {episode}/{self.episodes}')
            s = self.maze.start_pos
            a = self.greedy_action(s)
            for step in range(self.steps):
                # take action a, get new state and a reward r
                r, s_prime, terminate = self.maze.get_next_state_reward(s[0], s[1], a)
                a_prime = self.greedy_action(s_prime)
                self.Q[s[0], s[1], a] = self.Q[s[0], s[1], a] + self.alpha * (r + self.gamma * self.Q[s_prime[0], s_prime[1], a_prime] - self.Q[s[0], s[1], a])
                s = np.copy(s_prime)
                a = a_prime

                episode_reward += r

                if terminate: break # s is now terminal

            self.episode_rewards.append(episode_reward / (step + 1))

        for iy, ix in np.ndindex(self.pi.shape):
            self.pi[iy, ix] = np.argmax(self.Q[iy, ix])

        return self.pi

class QLearning:
    def __init__(self, maze, gamma, alpha, epsilon):
        '''
            gamma = discount factor in the reward; discounts future reward and guarantees convergence
            alpha = TD learning rate. The factor which the TD error gets multiplied with
            epsilon = epsilon for the epsilon-greedy step
        '''
        self.maze = maze
        self.gamma, self.alpha, self.epsilon = gamma, alpha, epsilon
        self.episodes = 1000
        self.steps = 1000
        self.initialize_policy_attrs()

    def initialize_policy_attrs(self):
        # Initialize policy parameters
        self.n_actions = len(self.maze.actions)
        # Initialize Q matrix for all (state, action) pairs
        self.Q = np.zeros((*self.maze.data.shape, self.n_actions))
        # Action Space
        # Up = 0, Down = 1, Left = 2, Right = 3
        self.pi = np.ones_like(self.maze.data) * 2
        # self.rewards = np.zeros((self.episodes, self.steps))
        self.episode_rewards = []

    def greedy_action(self, s):
        # draw an epsilon greedy action
        # generate a random uniform number
        toss = np.random.random()
        if toss <= 1 - self.epsilon: # greedy choice, a = argmax(Q_i)
            action = np.argmax(self.Q[s[0], s[1]])
        else:
            action = np.random.randint(0, self.n_actions) # random choice
        return action

    def learn_policy(self):
        for episode in range(self.episodes):
            episode_reward = 0
            # Print every 100 episodes
            if (episode % 100 == 0): print(f'Episode {episode}/{self.episodes}')
            s = self.maze.start_pos
            a = self.greedy_action(s)
            for step in range(self.steps):
                # take action a, get new state and a reward r
                r, s_prime, terminate = self.maze.get_next_state_reward(s[0], s[1], a)
                #a_prime = self.greedy_action(s_prime)
                #a_prime = np.argmax(self.Q[s[0], s[1]])
                # todo: have this as a function
                a_prime = np.random.choice(np.where(self.Q[s_prime[0], s_prime[1]] == np.amax(self.Q[s_prime[0], s_prime[1]]))[0])
                self.Q[s[0], s[1], a] = self.Q[s[0], s[1], a] + self.alpha * (r + self.gamma * self.Q[s_prime[0], s_prime[1], a_prime] - self.Q[s[0], s[1], a])
                s = np.copy(s_prime)

                episode_reward += r

                if terminate: break # s is now terminal

            self.episode_rewards.append(episode_reward / (step + 1))

        for iy, ix in np.ndindex(self.pi.shape):
            self.pi[iy, ix] = np.argmax(self.Q[iy, ix])

        return self.pi
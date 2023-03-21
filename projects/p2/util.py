import numpy as np
from matplotlib import pyplot as plt, colors, animation, patches

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

class Agent:
    def __init__(self, maze, policy):
        self.policy = policy
        self.maze = maze
        self.initialize_maze_attrs()
        self.max_steps = 100

    def initialize_maze_attrs(self):
        self.reward_tot = 0
        self.reward_hist=[]
        # Initialize path
        self.curr_pos = self.maze.start_pos
        self.path = []
        self.path.append(self.curr_pos)

    def learn_policy(self):
        # TODO: The learn policy is actually happening inside the policy, this is more of a use policy and transition probs to actually walk the agent
        policy = self.policy.learn_policy()

        decoded_state = self.maze.decode_state(self.curr_pos)
        # Continue transitions and moving until the final goal state has been reached, or if max iterations pass
        iter_num = 0
        while decoded_state != 'goal' and iter_num < self.max_steps:
            action = policy[self.curr_pos[0], self.curr_pos[1]]
            action = self.step_randomizer(action)
            self.step(action)
            decoded_state = self.maze.decode_state(self.curr_pos)
            iter_num += 1

    def step_randomizer(self, action):
        '''
        # An action has been chosen by the policy, so here we account for transition probabilities
        # The outcome is that the action can be altered/randomized when the agent is attempting to follow it

        If action right (R) is selected, the agent:
            moves to the state in right with probability 1 − p
            moves to the state in left with probability p/3
            moves to the the state in up with probability p/3
            moves to the state in down with probability p/3

        if any of the neighboring cells are wall, the agent stays in the current cell.
        '''
        return np.random.choice(self.maze.actions, replace=False, p=self.maze.transition_probs[action])

    def step(self, action):
        assert action in self.maze.action_space.keys()
        maze = self.maze.data
        # Determine what the updated position would be based on the action that has been chosen
        updated_pos = self.curr_pos + self.maze.action_space[action]
        decoded_state = self.maze.decode_state(updated_pos)
        # Update the position of the agent only if the new state is NOT a wall. This simulates the agent hitting a wall in remaining in its current state.
        if decoded_state != 'wall': self.curr_pos = updated_pos
        # Update agent's path + accumulate reward
        self.path.append(self.curr_pos)
        self.reward_tot += self.maze.get_reward(self.curr_pos)
        self.reward_hist.append(self.reward_tot)


class Maze:
    def __init__(self, maze_file, start_pos, transition_randomness=0.0):
        self.data = np.loadtxt(maze_file)
        self.start_pos = np.array(start_pos)

        # State + Reward Encodings 
        self.action_cost = - 1
        self.state_encodings = {'wall': 1, 'free': 0, 'bump': 2, 'oil': 3, 'goal': 5} # ensure everything is unique
        self.rewards_encodings = {'free': 0, 'bump': -10, 'oil': -5, 'goal': 200}
        # For simplicity later, we create an inverse dictionary of the state_encodings. This prevents performing inverse lookups every time a state needs to be decoded
        self.state_decodings = {v: k for k, v in self.state_encodings.items()}
        for i in self.rewards_encodings.keys(): self.rewards_encodings[i] += self.action_cost
        # Action Space
        # Up = 0, Down = 1, Left = 2, Right = 3
        self.action_space = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        self.actions = list(self.action_space.keys()) # simple list of actions
        # TODO: Add checks on maze data validity
        # State Space, helps when initializing policies separately. All states that aren't a wall are valid.
        # This is a 2D array - first indexes rows and second indexes cols of the maze
        # self.state_space_idx = np.where(self.data != self.state_encodings['wall'])
        # self.state_space = self.data[self.state_space_idx]

        # Pre-populating transition probabilities for a given action, makes the step_randomizer function faster since these no longer have to be recomputed every time
        self.transition_randomness = transition_randomness
        self.transition_probs = {}
        for action in self.actions: self.transition_probs[action] = [1-self.transition_randomness if maze_action == action else self.transition_randomness / 3 for maze_action in self.actions ]

    def decode_state(self, pos):
        maze = self.data
        state = maze[pos[0], pos[1]]
        return self.state_decodings[int(state)]

    def get_reward(self, pos):
        # Returns the reward generated at position `pos`. In other words, get_reward(pos) = r(s = pos)
        # Special case: If the position is a wall, agent should get reward for the current state
        decoded_state = self.decode_state(pos)
        return self.rewards_encodings[decoded_state]

    def get_next_states(self, x, y, action, verbose=True):
        '''
            Returns a list of 3-tuples: (s', r, p)
            s' = (x', y') of the new state
        '''
        curr_pos = np.array([x, y])
        curr_state = self.data[x, y]
        next_states = []

        if (self.state_decodings[curr_state] != 'wall'): # current state is valid
            # given action, there are 4-possible next states, each having a different transition probability depending on the action iteself
            probs = self.transition_probs[action]
            for idx in range(len(probs)): #idx is the same for probs as well as actions
                p = probs[idx]
                x_new, y_new = curr_pos + self.action_space[idx]
                new_state = self.data[x_new, y_new]
                if (self.state_decodings[new_state] == 'wall'): x_new, y_new = curr_pos
                new_pos = np.array([x_new, y_new])
                r = self.get_reward(new_pos)
                # if (self.state_decodings[new_state] == 'goal'): r = 0
                next_states.append((new_pos, r, p))
        return next_states


    # Functions for visualization
    def animate(self, agent):
        # Given an agent, animate the path it explored
        path = agent.path
        reward_hist = agent.reward_hist

        fig, ax = self.draw(display=False)
        # initializing empty values
        # for x and y co-ordinates
        xdata, ydata = [], [] 

        # create a circle to denote position of agent
        center = (path[0][1]+0.5,path[0][0]+0.5)
        agent = patches.Circle(center, radius = 0.2)
        agent.set(alpha = 0.40)
        ax.add_patch(agent)

        # text to show steps and reward
        reward_str = ''
        reward_txt = ax.text(0.5, 1, reward_str, bbox={'facecolor': 'white'}, verticalalignment='top', horizontalalignment='center', transform=ax.transAxes,)

        # create a line to trace path taken
        line, = ax.plot([], [], '--', lw=1, ms=2)
        def init(): 
            line.set_data([], []) 
            return line,

        # animation function 
        def animate(i): 
            xdata.append(path[i][1]+0.5) 
            ydata.append(path[i][0]+0.5) 
            line.set_data(xdata, ydata) 
            agent.set(center=(xdata[i],ydata[i]))
            agent.set(alpha=1)
            if i < len(reward_hist): reward_str=(f'step: {i}, reward: {reward_hist[i]}')
            else: reward_str=(f'step: {i}, reward: {reward_hist[i-1]}, finished')
            reward_txt.set_text(reward_str)
            return line,agent,reward_txt

        # calling the animation function     
        anim = animation.FuncAnimation(fig, animate,  init_func = init, frames = len(path), interval = 50, blit = True, repeat=False) 
        self.display()

    def draw(self, display=True, V=None, pi=None):
        # Draw the grid using pyplot
        fig, ax = plt.subplots(figsize=(10, 8))
        colormap = colors.ListedColormap(["white","black","orange","red","green"])
        data = np.flipud(self.data)
        ax.invert_yaxis()
        plt.pcolormesh(data[::-1], cmap=colormap, edgecolors='k', linewidths=0.1)
        ax.axis('off')
        patch_list = [ patches.Patch(edgecolor='k',facecolor='red', label='oil',),
                    patches.Patch(edgecolor='k',facecolor='black', label='wall',),
                    patches.Patch(edgecolor='k',facecolor='white', label='free space',),
                    patches.Patch(edgecolor='k',facecolor='orange', label='bump',),
                    patches.Patch(edgecolor='k',facecolor='green', label='end point',),
                    patches.Patch(edgecolor='k',facecolor='blue', label='start',),]
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=patch_list, fancybox=True, shadow=True)
        
        if (V is not None):
            for iy, ix in np.ndindex(self.data.shape):
                val = V[iy, ix]        
                ax.text(ix + 0.5, iy + 0.5, str(int(val)), horizontalalignment='center', verticalalignment='center')
        elif (pi is not None):
            for iy, ix in np.ndindex(self.data.shape):
                dy, dx = self.action_space[pi[iy, ix]]
                ax.arrow(ix + 0.5, iy + 0.5, dx / 4, dy / 4, head_width=0.4, head_length=0.2, fc='k', ec='k')
        
        if (display): self.display()
        return fig, ax

    # Generic display function
    def display(self):
        # Display the plot
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(1)
        print('Press enter to close the plot')
        input()
        plt.close()
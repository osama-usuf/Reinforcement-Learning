import numpy as np
from matplotlib import pyplot as plt, colors, animation, patches

class RandomPolicy:
    def __init__(self, maze, num_steps):
        self.maze = maze
        self.num_steps = num_steps

    def step_rand_dir(self):
        # Pick an action randomly
        return np.random.choice(self.maze.actions, replace=False)

    def generate_trajectory(self):
        # Simply generates a list of random actions
        # These do not account for transition probabilities
        trajectory = []
        for i in range(self.num_steps): trajectory.append(self.step_rand_dir())
        return trajectory

class PolicyIteration:
    def __init__(self, maze, transition_randomness, gamma, theta):
        '''
            transition_randomness = p. Agent moves to indended state with 1 - p probability, and a random neighbor with p / 3 probability.
            gamma = discount factor in the reward; discounts future reward and guarantees convergence
            theta = a small +ve no. that determines the accuracy of estimation of the policy during evaluation
        '''
        self.maze = maze
        self.transition_randomness, gamma, theta = transition_randomness, gamma, theta
        self.initialize()

    def initialize(self):
        # Initialize policy parameters
        # Firstly, we need a vector of valid states from the maze object
        n = len(self.maze.state_space)
        self.V = np.zeros(n)
        # Action Space
        # Up = 0, Down = 1, Left = 2, Right = 3
        self.pi = np.ones(n) * 2

    def generate_trajectory(self):
        # Simply generates a list of random actions
        # These do not account for transition probabilities
        trajectory = []
        for i in range(self.num_steps): trajectory.append(self.step_rand_dir())
        return trajectory

class Agent:
    def __init__(self, maze, policy, transition_randomness=0.5):
        self.policy = policy
        self.maze = maze
        self.initialize_maze_attrs()
        self.transition_randomness = transition_randomness
        # Pre-populating transition probabilities for a given action, makes the step_randomizer function faster since these no longer have to be recomputed every time
        self.transition_probs = {}
        for action in self.maze.actions: self.transition_probs[action] = [1-self.transition_randomness if maze_action == action else self.transition_randomness / 3 for maze_action in self.maze.actions ]

    def initialize_maze_attrs(self):
        self.reward_tot = 0
        self.reward_hist=[]
        # Initialize path
        self.curr_pos = self.maze.start_pos
        self.path = []
        self.path.append(self.curr_pos)

    def learn_policy(self):
        # TODO: The learn policy is actually happening inside the policy, this is more of a use policy and transition probs to actually walk the agent
        trajectory = self.policy.generate_trajectory()
        for action in trajectory:
            action = self.step_randomizer(action)
            self.step(action)

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
        return np.random.choice(self.maze.actions, replace=False, p=self.transition_probs[action])

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
        self.accumulate_reward()

    def accumulate_reward(self):
        self.reward_tot += self.maze.get_reward(self.curr_pos)
        self.reward_hist.append(self.reward_tot)


class Maze:
    def __init__(self, maze_file, start_pos):
        self.data = np.loadtxt(maze_file)
        self.start_pos = np.array(start_pos)

        # State + Reward Encodings 
        self.action_cost = - 1
        self.state_encodings = {'wall': 1, 'free': 0, 'bump': 2, 'oil': 3, 'goal': 5} # ensure everything is unique
        self.rewards_encodings = {'wall': 0, 'free': 0, 'bump': -10, 'oil': -5, 'goal': 200}
        # For simplicity later, we create an inverse dictionary of the state_encodings. This prevents performing inverse lookups every time a state needs to be decoded
        self.state_decodings = {v: k for k, v in self.state_encodings.items()}
        for i in self.rewards_encodings.keys(): self.rewards_encodings[i] += self.action_cost
        # Action Space
        # Up = 0, Down = 1, Left = 2, Right = 3
        self.action_space = {0: [1, 0], 1: [-1, 0], 2: [0, -1], 3: [0, 1]}
        self.actions = list(self.action_space.keys()) # simple list of actions
        # TODO: Add checks on maze data validity
        # State Space, helps when initializing policies separately. All states that aren't a wall are valid.
        # This is a 2D array - first indexes rows and second indexes cols of the maze
        self.state_space_idx = np.where(self.data != self.state_encodings['wall'])
        self.state_space = self.data[self.state_space_idx]

    def decode_state(self, pos):
        maze = self.data
        state = maze[pos[0], pos[1]]
        return self.state_decodings[int(state)]

    def get_reward(self, pos):
        # Returns the reward generated at position `pos`. In other words, get_reward(pos) = r(s = pos)
        decoded_state = self.decode_state(pos)
        return self.rewards_encodings[decoded_state]

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
        fig, ax = plt.subplots()
        colormap = colors.ListedColormap(["white","black","orange","red","green"])
        data = np.flipud(self.data)
        ax.invert_yaxis()
        plt.pcolormesh(data[::-1], cmap=colormap, edgecolors='k', linewidths=0.1)
        ax.axis('off')
        red_patch = patches.Patch(edgecolor='k',facecolor='red', label='oil',)
        blk_patch = patches.Patch(edgecolor='k',facecolor='black', label='wall',)
        wht_patch = patches.Patch(edgecolor='k',facecolor='white', label='free space',)
        org_patch = patches.Patch(edgecolor='k',facecolor='orange', label='bump',)
        grn_patch = patches.Patch(edgecolor='k',facecolor='green', label='end point',)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)
                ,handles=[red_patch,blk_patch,wht_patch,org_patch,grn_patch]
                ,fancybox=True,shadow=True)
        
        if (V is not None):
            assert len(V) == len(self.state_space)
            for idx in range(len(V)):                
                x, y = self.state_space_idx[1][idx], self.state_space_idx[0][idx]
                val = V[idx]
                dy, dx = self.action_space[val]
                ax.text(x, y + 0.5, str(val // 1))
        elif (pi is not None):
            assert len(pi) == len(self.state_space)
            for idx in range(len(pi)):                
                x, y = self.state_space_idx[1][idx], self.state_space_idx[0][idx]
                dy, dx = self.action_space[pi[idx]]
                ax.arrow(x + 0.5, y + 0.5, dx / 4, -1*dy / 4, head_width=0.4, head_length=0.2, fc='k', ec='k')

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
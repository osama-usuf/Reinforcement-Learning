import numpy as np
from matplotlib import pyplot as plt, colors, animation, patches

class Agent:
    def __init__(self, maze):
        self.maze = maze
        self.initialize_maze_attrs()

    def initialize_maze_attrs(self):
        self.reward_tot = 0
        self.reward_hist=[]
        # Initialize path
        self.curr_pos = self.maze.start_pos
        self.path = []
        self.path.append(self.curr_pos)

    def step_rand_dir(self):
        sample = np.random.random()

        if sample <= 0.25:
            direction = 'up'
        elif 0.25 < sample <= 0.5:
            direction = 'left'
        elif 0.5 < sample <= 0.75:
            direction = 'down'
        elif 0.75 < sample <= 1:
            direction = 'right'

        direction = self.step_randomizer(direction)
        self.step(direction)

    def step(self, direction):
        maze = self.maze.data
        if direction == 'up':
            update = self.curr_pos + [-1,0]
            if maze[[update[0]],[update[1]]] == 1: # wall
                self.curr_pos = self.curr_pos
                self.path.append(self.curr_pos)
                self.reward_check()
            elif maze[[update[0]],[update[1]]] != 1:
                self.curr_pos = update
                self.path.append(self.curr_pos)
                self.reward_check()

        elif direction == 'left':
            update = self.curr_pos + [0,-1]
            if maze[[update[0]],[update[1]]] == 1:
                self.curr_pos = self.curr_pos
                self.path.append(self.curr_pos)
                self.reward_check()
            elif maze[[update[0]],[update[1]]] != 1:
                self.curr_pos = update
                self.path.append(self.curr_pos)
                self.reward_check()

        elif direction == 'down':
            update = self.curr_pos + [1,0]
            if maze[[update[0]],[update[1]]] == 1:
                self.curr_pos = self.curr_pos
                self.path.append(self.curr_pos)
                self.reward_check()
            elif maze[[update[0]],[update[1]]] != 1:
                self.curr_pos = update
                self.path.append(self.curr_pos)
                self.reward_check()

        elif direction == 'right':
            update = self.curr_pos + [0,1]
            if maze[[update[0]],[update[1]]] == 1:
                self.curr_pos = self.curr_pos
                self.path.append(self.curr_pos)
                self.reward_check()
            elif maze[[update[0]],[update[1]]] != 1:
                self.curr_pos = update
                self.path.append(self.curr_pos)
                self.reward_check()

        else:
            print("Invalid direction used")

    def step_randomizer(self, direction):

        random_param = 0.5
        random_prob = np.random.random_sample()

        if random_prob > (1-random_param):
            return direction
        else:
            # print("random step has occured")
            if direction == 'up':
                if random_prob <= ((1-random_param)/3):
                    direction = 'left'
                    return direction
                if ((1-random_param)/3) < random_prob <= (2*(1-random_param)/3):
                    direction = 'right'
                    return direction
                if (2*(1-random_param)/3) < random_prob <= (1-random_param):
                    direction = 'down'
                    return direction
            elif direction == 'left':
                if random_prob <= ((1-random_param)/3):
                    direction = 'up'
                    return direction
                if ((1-random_param)/3) < random_prob <= (2*(1-random_param)/3):
                    direction = 'right'
                    return direction
                if (2*(1-random_param)/3) < random_prob <= (1-random_param):
                    direction = 'down'
                    return direction
            elif direction == 'right':
                if random_prob <= ((1-random_param)/3):
                    direction = 'up'
                    return direction
                if ((1-random_param)/3) < random_prob <= (2*(1-random_param)/3):
                    direction = 'left'
                    return direction
                if (2*(1-random_param)/3) < random_prob <= (1-random_param):
                    direction = 'down'
                    return direction
            elif direction == 'down':
                if random_prob <= ((1-random_param)/3):
                    direction = 'up'
                    return direction
                if ((1-random_param)/3) < random_prob <= (2*(1-random_param)/3):
                    direction = 'right'
                    return direction
                if (2*(1-random_param)/3) < random_prob <= (1-random_param):
                    direction = 'left'
                    return direction

    def reward_check(self):
        maze = self.maze.data

        action = -1
        oil = -5
        bump = -10
        goal = 200

        if maze[[self.curr_pos[0]],[self.curr_pos[1]]] == 2:
            self.reward_tot += bump
        if maze[[self.curr_pos[0]],[self.curr_pos[1]]] == 3:
            self.reward_tot += oil
        if maze[[self.curr_pos[0]],[self.curr_pos[1]]] == 5:
            self.reward_tot += goal
        self.reward_tot += action
        self.reward_hist.append(self.reward_tot)

    def generate_rand_path(self,num_steps):
        for i in range(num_steps):
            self.step_rand_dir()


class Maze:
    def __init__(self, maze_file, start_pos):
        self.data = np.loadtxt(maze_file)
        self.start_pos = np.array(start_pos)

        # State - Reward Encodings
        '''
        1 = walls
        0 = free path
        2 = bump -> -10
        3 = oil -> -5
        5 = end -> +200
        '''
        # Todo - add checks on maze data validity

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
        agent = patches.Circle(center,radius = 0.2)
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

    def draw(self, display=True):
        # Draw the grid using pyplot
        fig, ax = plt.subplots()
        colormap = colors.ListedColormap(["white","black","orange","red","green"])

        data = np.flipud(self.data)
        ax.invert_yaxis()
        plt.pcolormesh(data[::-1],cmap=colormap,edgecolors='k', linewidths=0.1)
        ax.axis('off')
        red_patch = patches.Patch(edgecolor='k',facecolor='red', label='oil',)
        blk_patch = patches.Patch(edgecolor='k',facecolor='black', label='wall',)
        wht_patch = patches.Patch(edgecolor='k',facecolor='white', label='free space',)
        org_patch = patches.Patch(edgecolor='k',facecolor='orange', label='bump',)
        grn_patch = patches.Patch(edgecolor='k',facecolor='green', label='end point',)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)
                ,handles=[red_patch,blk_patch,wht_patch,org_patch,grn_patch]
                ,fancybox=True,shadow=True)
        if (display): self.display()
        return fig, ax
        
    def display(self):
        # Display the plot
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(1)
        print('Press enter to close the plot')
        input()
        plt.close()
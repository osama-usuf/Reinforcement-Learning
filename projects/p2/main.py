from util import Agent, Maze
import matplotlib.pyplot as plt

num_step = 50
num_run = 500

maze = Maze(maze_file='mazes/base.txt', start_pos=[15, 4])

data = []

for i in range(num_run):
    agent = Agent(maze)
    agent.generate_rand_path(num_steps=num_step)
    data.append(agent.reward_tot)

plt.hist(x=data, bins=50)
plt.title(f'Reward Distribution of {num_run} Independent Runs')

# Animate the agent. Last path + reward history is used.
maze.animate(agent)
# Draw the maze
maze.draw()
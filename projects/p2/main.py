from util import Agent, Maze, RandomPolicy, PolicyIteration
import matplotlib.pyplot as plt

# Load maze object at specified starting position.
maze = Maze(maze_file='mazes/base.txt', start_pos=[15, 4])

# Experiment 1 - Random Policy
# Simulates an agent randomly navigating the maze num_steps times, for a total of 500 independent runs

exp_1 = False
if (exp_1):
    num_run = 10
    data = []
    policy = RandomPolicy(maze=maze, num_steps=47)
    for i in range(num_run):
        agent = Agent(maze, policy)
        agent.learn_policy()
        data.append(agent.reward_tot)

    plt.hist(x=data, bins=50)
    plt.title(f'Reward Distribution of {num_run} Independent Runs')

    # Animate the agent. The last random trajectory from the experiment is used.
    maze.animate(agent)
    # Draw the maze, empty
    # maze.draw()

# Experiment 2 - Vector Form Policy Iteration

exp_2 = True
if (exp_2):
    # Scenario a - Base Scenario
    p = 0
    gamma = 0.95
    theta = 0.01 
    
    maze = Maze(maze_file='mazes/base.txt', start_pos=[15, 4], transition_randomness=p)
    policy = PolicyIteration(maze=maze, transition_randomness=p, gamma=gamma, theta=theta)
    agent = Agent(maze, policy)
    agent.learn_policy()
    # Visualize Learned Policy
    maze.draw(display=True, V=None, pi=policy.pi)
    # print(policy.V)
    # maze.draw(display=True, V=policy.V, pi=None)

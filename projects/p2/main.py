from util import Agent, Maze
from policies import RandomPolicy, PolicyIteration, ValueIteration
import matplotlib.pyplot as plt
import time

# Load maze object at specified starting position.
maze = Maze(maze_file='mazes/base.txt', start_pos=[15, 4])

# Draw the base maze
# maze.draw(display=True, V=None, pi=None)

# Experiment 1 - Random Policy
# Simulates an agent randomly navigating the maze num_steps times

exp_1 = False
if (exp_1):
    policy = RandomPolicy(maze=maze)
    agent = Agent(maze, policy)
    agent.learn_policy()

    # Visualize Learned Policy
    maze.draw(display=True, V=None, pi=policy.pi)

    # Animate the agent. The last random trajectory from the experiment is used.
    maze.animate(agent)
    # Draw the maze, empty
    # maze.draw()

# Experiment 2 - Vector Form Policy Iteration

exp_2 = True
if (exp_2):

    # # Manual Policy
    # # left left up right right down
    # # agent.step(2)
    # # agent.step(2)
    # # agent.step(0)
    # # agent.step(3)
    # # agent.step(3)
    # # agent.step(1)
    # # maze.animate(agent)

    # a - Base Scenario
    '''
        Stochasticity is low, so once the final state is reached, there's a very high likelihood (98%) that
        the agent remains in the goal state (and keeps getting the corresponding reward). 
        This is why the V values are really high in this scenario.
    '''
    p = 0.02
    gamma = 0.95
    theta = 0.01
    
    maze = Maze(maze_file='mazes/base.txt', start_pos=[15, 4], transition_randomness=p)
    policy = PolicyIteration(maze=maze, gamma=gamma, theta=theta)
    agent = Agent(maze, policy)

    start = time.time()
    agent.learn_policy()
    end = time.time()
    print(end - start)
    # Visualize Learned Policy
    maze.draw(display=True, V=None, pi=policy.pi)
    maze.draw(display=True, V=policy.V, pi=None)
    agent.follow_policy(optimal=True)
    maze.animate(agent)
    agent.follow_policy(optimal=False)
    maze.animate(agent)

    # b - Large Stochasticity
    p = 0.5
    gamma = 0.95
    theta = 0.01
    
    maze = Maze(maze_file='mazes/base.txt', start_pos=[15, 4], transition_randomness=p)
    policy = PolicyIteration(maze=maze, gamma=gamma, theta=theta)
    agent = Agent(maze, policy)

    start = time.time()
    agent.learn_policy()
    end = time.time()
    print(end - start)
    # Visualize Learned Policy
    maze.draw(display=True, V=None, pi=policy.pi)
    maze.draw(display=True, V=policy.V, pi=None)
    agent.follow_policy(optimal=True)
    maze.animate(agent)
    agent.follow_policy(optimal=False)
    maze.animate(agent)

    # c - Small Discount Factor
    p = 0.02
    gamma = 0.55
    theta = 0.01
    
    maze = Maze(maze_file='mazes/base.txt', start_pos=[15, 4], transition_randomness=p)
    policy = PolicyIteration(maze=maze, gamma=gamma, theta=theta)
    agent = Agent(maze, policy)

    start = time.time()
    agent.learn_policy()
    end = time.time()
    print(end - start)
    # Visualize Learned Policy
    maze.draw(display=True, V=None, pi=policy.pi)
    maze.draw(display=True, V=policy.V, pi=None)
    agent.follow_policy(optimal=True)
    maze.animate(agent)
    agent.follow_policy(optimal=False)
    maze.animate(agent)


exp_3 = True
if (exp_3):
    # Scenario a - Base Scenario
    p = 0.02
    gamma = 0.95
    theta = 0.01
    
    maze = Maze(maze_file='mazes/base.txt', start_pos=[15, 4], transition_randomness=p)
    policy = ValueIteration(maze=maze, gamma=gamma, theta=theta)
    agent = Agent(maze, policy)

    start = time.time()
    agent.learn_policy()
    end = time.time()
    print(end - start)
    # Visualize Learned Policy
    maze.draw(display=True, V=None, pi=policy.pi)
    maze.draw(display=True, V=policy.V, pi=None)
    agent.follow_policy(optimal=True)
    maze.animate(agent)
    agent.follow_policy(optimal=False)
    maze.animate(agent)

    # b - Large Stochasticity
    p = 0.5
    gamma = 0.95
    theta = 0.01
    
    maze = Maze(maze_file='mazes/base.txt', start_pos=[15, 4], transition_randomness=p)
    policy = ValueIteration(maze=maze, gamma=gamma, theta=theta)
    agent = Agent(maze, policy)

    start = time.time()
    agent.learn_policy()
    end = time.time()
    print(end - start)
    # Visualize Learned Policy
    maze.draw(display=True, V=None, pi=policy.pi)
    maze.draw(display=True, V=policy.V, pi=None)
    agent.follow_policy(optimal=True)
    maze.animate(agent)
    agent.follow_policy(optimal=False)
    maze.animate(agent)

    # c - Small Discount Factor
    p = 0.02
    gamma = 0.55
    theta = 0.01
    
    maze = Maze(maze_file='mazes/base.txt', start_pos=[15, 4], transition_randomness=p)
    policy = ValueIteration(maze=maze, gamma=gamma, theta=theta)
    agent = Agent(maze, policy)

    start = time.time()
    agent.learn_policy()
    end = time.time()
    print(end - start)
    # Visualize Learned Policy
    maze.draw(display=True, V=None, pi=policy.pi)
    maze.draw(display=True, V=policy.V, pi=None)
    agent.follow_policy(optimal=True)
    maze.animate(agent)
    agent.follow_policy(optimal=False)
    maze.animate(agent)
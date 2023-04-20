from util import Agent, Maze
from policies import RandomPolicy, PolicyIteration, ValueIteration, SARSA, QLearning
import matplotlib.pyplot as plt
import time
import random
import numpy as np

'''
Project 2 - Experiments 1, 2, 3
Project 3 - Experiments 4, 5
'''

# Project 2
# Experiment 1 - Random Policy
# Simulates an agent randomly navigating the maze num_steps times

exp_1 = False
if (exp_1):
    # Load maze object at specified starting position.
    maze = Maze(maze_file='mazes/base.txt', start_pos=[15, 4])
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

exp_2 = False
if (exp_2):

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


exp_3 = False
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


# Project 3
seed = 1
random.seed(seed)
np.random.seed(seed)

p = 0.02
gamma = 0.95
alpha = 0.3
epsilon = 0.1
episodes = 1000
episode_steps = 1000
policy_steps = 500
runs = 10

exp_4 = True
exp_5 = True

if (exp_4):
    # SARSA algorithm
    print('='*40)
    print('SARSA Algorithm')
    print('='*40)

    goals_found = 0

    run_to_plot = 10
    for run in range(runs):
        print(f'\nIndependent run {run+1}/{runs}')
        maze = Maze(maze_file='mazes/base.txt', start_pos=[15, 4], transition_randomness=p)
        policy = SARSA(maze=maze, gamma=gamma, alpha=alpha, epsilon=epsilon, episodes=episodes, steps=steps)
        agent = Agent(maze, policy, policy_steps)
        agent.learn_policy()
        print(f'Learning Terminated')
        goal_found = agent.follow_policy(optimal=True)
        if goal_found: 
            goals_found += 1
            print('Goal state has been found')
        else:
            print('Goal state NOT found')

        np.savetxt(f'data/SARSA_{run}.txt', policy.episode_rewards)
        
        # Visualize Learned Policy
        if (run + 1 == run_to_plot):
            maze.draw(display=True, V=None, pi=policy.pi)
            agent.follow_policy(optimal=True)
            maze.animate(agent)
    print(f'{goals_found}/{runs} goal states found!')

if (exp_5):
    # QLearning algorithm
    print('='*40)
    print('QLearning Algorithm')
    print('='*40)

    goals_found = 0

    run_to_plot = 1
    for run in range(runs):
        print(f'\nIndependent run {run+1}/{runs}')
        maze = Maze(maze_file='mazes/base.txt', start_pos=[15, 4], transition_randomness=p)
        policy = QLearning(maze=maze, gamma=gamma, alpha=alpha, epsilon=epsilon, episodes=episodes, steps=steps)
        agent = Agent(maze, policy, policy_steps)
        agent.learn_policy()
        print(f'Learning Terminated')
        goal_found = agent.follow_policy(optimal=True)
        if goal_found: 
            goals_found += 1
            print('Goal state has been found')
        else:
            print('Goal state NOT found')

        np.savetxt(f'data/QLearning_{run}.txt', policy.episode_rewards)
        
        # Visualize Learned Policy
        if (run + 1 == run_to_plot):
            maze.draw(display=True, V=None, pi=policy.pi)
            agent.follow_policy(optimal=True)
            maze.animate(agent)
    print(f'{goals_found}/{runs} goal states found!')
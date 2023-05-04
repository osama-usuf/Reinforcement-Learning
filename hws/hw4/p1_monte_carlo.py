import numpy as np
import random

# State A = 0
# State B = 1
# Action a1 = 0
# Action a2 = 1
state_encodings = {0: 'A', 1: 'B'}
action_encodings = {0: 'a1', 1: 'a2'}

def get_reward(s, a, s_prime):
    r = 0
    if (s_prime == 1):
        r += 5
    elif (s_prime == 0):
        r += 0
    if (a == 0): 
        r += 0
    elif (a == 1):
        r += -1
    return r

def get_next_state(s, action):
    # a1 - stay in same state
    if (action == 0):
        return s
    # a2 - switch state
    elif  (action == 1):
        if (s==0): return 1
        elif (s==1): return 0

def greedy_action(s):
    # draw an epsilon greedy action
    # generate a random uniform number
    toss = np.random.random()
    if toss <= 1 - epsilon:
        action = pi[s]
    else:
        action = np.random.randint(0, 2) # random choice
    return action

def generate_episode():
    s = 0 # TODO: Does initial state need to change?
    episode = []
    while len(episode) < episode_length:
        a = greedy_action(s)
        s_prime = get_next_state(s, a)
        r = get_reward(s, a, s_prime)
        episode.append((s, a, s_prime, r))
        s = s_prime
    return episode

def convert_episode(episode):
    friendly_ep = []
    for s, a, s_prime, r in episode:
        friendly_ep.append((state_encodings[s], action_encodings[a], state_encodings[s_prime], r))
    return friendly_ep

def monte_carlo_policy():
    for iters in range(max_iters):
        episode = generate_episode()
        friendly_episode = convert_episode(episode)
        print('=' * 100)
        print(f'Episode {iters+1}: {episode}')
        print(f'Episode {iters+1}: {friendly_episode}')
        # Update Q Values
        for i in range(len(episode)):
            s, a, s_prime, r = episode[i]
            print(f'\nFor state-action pair ({s}, {a})')
            R = r
            power = 1
            for j in range(i+1, len(episode)):
                R += episode[j][3] * (gamma ** power)
                power += 1
            returns[s][a].append(R)
            Q[s][a] = np.average(returns[s][a])
            print('returns', returns)
            print('Q', Q)

        # Update policy
        curr_policy = pi.copy()
        for k in range(len(episode)):
            s, _, _, _ = episode[k]
            pi[s] = np.argmax(Q[s])
        print('\nFinal Q', Q)
        print('Final Returns', returns)
        print('Final Policy', pi)
        new_policy = pi.copy()

        if (new_policy == curr_policy and iters != 0):
            print('Policy unchanged b/w two consecutive iterations, ending.')
            break

if __name__ == '__main__':
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    pi = [0, 1] # initial policy = [a1, a2]
    Q = [[0, 0], [0, 0]]
    returns = [[[], []], [[], []]]
    gamma = 0.9
    episode_length = 5

    max_iters = 5
    epsilon = 0.4
    monte_carlo_policy()
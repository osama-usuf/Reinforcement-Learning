import numpy as np
import random

# State A = 0
# State B = 1
# Action a1 = 0
# Action a2 = 1

def update_policy():
    # at each state, you can technically 
    actions = [0, 1]
    states = [0, 1]
    threshold = 0.1
    for s in states:
        probs = []
        for i in actions:
            probs.append(np.exp(H[s][i]) / np.sum(np.exp(H[s])))
        assert np.sum(probs) - 1 < threshold
        pi[s] = probs


    # pi[s][a] = np.exp(H[s][a]) / np.sum(np.exp(H[s]))


def actor_critic_iteration():
    t = 0
    s = episode[t][0]
    while t < len(episode) :
        print('='*100)
        print(f'Timestep t = {t}')
        update_policy()
        print('Updated policy', pi)
        if (t == len(episode) - 1):
            break
        s, a, r_next = episode[t]
        s_next = episode[t+1][0]
        delta_t = r_next + gamma * V[s_next] - V[s]
        V[s] += alpha * delta_t
        H[s][a] += beta * delta_t * (1 - pi[s][a])
        print('V', V)
        print('Preferences', H)

        t += 1


if __name__ == '__main__':
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    V = [0, 0] # initial policy = [a1, a2]
    H = [[0, 0], [0, 0]]
    pi = [[None, None], [None, None]]
    # (s, a, r) tuples
    episode = [ (0, 0, 10), (0, 1, -5), (1, 0, 40), 
                (0, 1, -5), (1, 1, 20), (0, 0, 10), (0, None, None)]

    alpha = 0.5
    beta = 0.1
    gamma = 0.9

    max_iters = 5
    actor_critic_iteration()
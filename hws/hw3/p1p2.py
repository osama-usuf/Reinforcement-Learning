import numpy as np

states = [0, 1]
actions = [1, 2]

def p(s_prime, s, action):
    assert s in states
    assert action in actions
    if action == 1:
        if (s == 0 and s_prime == 0):
            return 1
        elif (s == 0 and s_prime == 1):
            return 0
        elif (s == 1 and s_prime == 0):
            return 0
        else:
            return 1
    else: # action == 2
        if (s == 0 and s_prime == 0):
            return 0
        elif (s == 0 and s_prime == 1):
            return 1
        elif (s == 1 and s_prime == 0):
            return 1
        else:
            return 0

def r(s, action, s_prime):
    reward = 0
    if s_prime == 1:
        reward += 1
    if s_prime == 0:
        reward += 0
    if action == 1 or action == 2:
        reward += 0
    return reward

def policy_iteration(V, pi, gamma, theta):
    max_loop_iters = 100
    max_outer_iters = 100
    iters = 0
    outer_iters = 0
    while outer_iters <= max_outer_iters:
        print('outer iteration', outer_iters)
        # policy evaluation
        while iters <= max_loop_iters:
            delta = 0
            for idx in range(len(V)):
                print('before', idx, V[idx])
                v = V[idx]
                val = 0
                for state_idx in range(len(states)):
                    val += p(states[state_idx], states[idx], pi[idx]) * (r(states[idx], pi[idx], states[state_idx]) + gamma * V[state_idx])
                V[idx] = val
                print('after', idx, V[idx])
                delta = max(delta, abs(v - V[idx]))
            iters += 1
            if (delta < theta): break

        # policy improvement
        policy_stable = True
        for s_idx in range(len(states)):
            old_action = pi[s_idx]
            action_outcomes = []
            for action in actions:
                reward = 0
                for s_prime_idx in range(len(states)): 
                    reward += p(states[s_prime_idx], states[s_idx], action) * (r(states[s_idx], action, states[s_prime_idx]) + gamma * V[s_prime_idx])
                action_outcomes.append(reward)
            max_action_idx = np.argmax(action_outcomes)
            pi[s_idx] = actions[max_action_idx]
            if (old_action != pi[s_idx]):
                policy_stable = False
        if (policy_stable):
            print('policy stable')
            return V, pi
            
        outer_iters += 1


def value_iteration(V, pi, gamma, theta):
    max_loop_iters = 100
    max_outer_iters = 100
    iters = 0

    while iters <= max_loop_iters:
        delta = 0
        print('Iteration', iters+1)

        for s_idx in range(len(states)):
            v = V[s_idx]
            action_outcomes = []
            for action in actions:
                reward = 0
                for s_prime_idx in range(len(states)): 
                    reward += p(states[s_prime_idx], states[s_idx], action) * (r(states[s_idx], action, states[s_prime_idx]) + gamma * V[s_prime_idx])
                action_outcomes.append(reward)
            max_action_idx = np.argmax(action_outcomes)
            V[s_idx] = action_outcomes[max_action_idx]

            delta = max(delta, abs(v-V[s_idx]))
        
        iters += 1

        print(V)

        if (delta < theta): 
            print('Value iteration converged')            
            break

    # Output policy
    for s_idx in range(len(states)):
        action_outcomes = []
        for action in actions:
            reward = 0
            for s_prime_idx in range(len(states)): 
                reward += p(states[s_prime_idx], states[s_idx], action) * (r(states[s_idx], action, states[s_prime_idx]) + gamma * V[s_prime_idx])
            action_outcomes.append(reward)
        max_action_idx = np.argmax(action_outcomes)

        pi[s_idx] = actions[max_action_idx]

    return V, pi
            

# Problem 1
V = [0, 0]
pi = [1, 1]
gamma = 0.9
theta = 0.85
# V, pi = policy_iteration(V, pi, gamma, theta)
# print(V, pi)

# Problem 2
V = [0, 0]
pi = [1, 1]
gamma = 0.9
theta = 0.85
V, pi = value_iteration(V, pi, gamma, theta)
print(V, pi)
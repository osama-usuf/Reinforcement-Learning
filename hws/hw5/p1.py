import numpy as np

def phi(s, a):
    return a**2 * s + a * s + a

def lspi_evaluate(w):
    for s in range(len(states)):
        for a in range(len(actions)):
            val = phi(states[s], actions[a]) * w
            Qs[s, a] = val
        # action selection
        pi[s] = actions[np.argmax(Qs[s])]
    print('Evaluation complete')
    print('Qs', Qs)
    print('pi', pi)

def lspi_improve():
    sum_A = 0
    sum_b = 0
    L = len(data)
    # Compute A
    for s, a, r, s_prime in data:
        s_prime_idx = states.index(s_prime)
        print('sum_A', phi(s, a) * (phi(s, a) - gamma * phi(s_prime, pi[s_prime_idx])))
        print('sum_B', phi(s, a) * r)
        sum_A += phi(s, a) * (phi(s, a) - gamma * phi(s_prime, pi[s_prime_idx]))
        sum_b += phi(s, a) * r
    A = np.array(sum_A / L)
    b = np.array(sum_b / L)
    A = A.reshape((k, k))
    b = b.reshape((k, 1))
    w = np.linalg.inv(A) * b
    print('Improvement complete')
    print('A', A)
    print('b', b)
    print('w', w)
    return w

if __name__ == "__main__":
    states = [-1, 1, 2]
    actions = [-1, 0, 1]
    gamma = 0.9
    Qs = np.zeros((len(states), len(actions)))
    pi = np.ones((len(states))) * -1
    data = [(1, 1, 1, 2), (2, 0, -1, 1), (1, -1, 0, -1)]
    k = 1 # no. of basis functions
    w = np.ones((k, 1))

    lspi_iters = 2

    lspi_evaluate(w)
    for i in range(lspi_iters):
        print('='*80)
        print(f'Iteration {i}')
        print('='*80)
        w = lspi_improve()
        lspi_evaluate(w)

    print('Final Qs', Qs)
    print('Final pi', pi)
    print('Final w', w)
import numpy as np

def phi0(s, a):
    return a**2 * s + a * s + a

def phi1(s, a):
    return a * s + a

def phi2(s, a):
    return a**2 * s

def lspi_evaluate(w):
    for s in range(len(states)):
        for a in range(len(actions)):
            val = 0
            for i in range(k):
                val += phis[i](states[s], actions[a]) * w[i]
            Qs[s, a] = val
        # action selection
        pi[s] = actions[np.argmax(Qs[s])]
    print('Evaluation complete')
    print('Qs', Qs)
    print('pi', pi)

def get_phi_vector(s, a):
    phi_evaluated = []
    for i in range(k):
        phi_evaluated.append(phis[i](s, a))
    phi_vector = np.array(phi_evaluated).reshape((k, 1))
    return phi_vector

def lspi_improve():
    L = len(data)
    A = np.zeros((k, k))
    b = np.zeros((k, 1))
    iteration = 1
    for s, a, r, s_prime in data:
        print(f'\ni={iteration}')
        s_prime_idx = states.index(s_prime)
        phi_s_a = get_phi_vector(s, a)
        phi_s_prime = get_phi_vector(s_prime, pi[s_prime_idx])
        A += phi_s_a @ (phi_s_a - gamma * phi_s_prime).T
        b += phi_s_a * r
        print('phi(s, a)', phi_s_a)
        print('phi(s_prime, pi[s_prime])', phi_s_prime)
        print('+= A', phi_s_a @ (phi_s_a - gamma * phi_s_prime).T)
        print('+= b', phi_s_a * r)
        iteration += 1

    A /= L
    b /= L
    w = np.linalg.inv(A) @ b
    print('Improvement complete')
    print('A', A)
    print('A inverse', np.linalg.inv(A))
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
    # For problem 1
    phis = [phi0]
    # For problem 2
    phis = [phi1, phi2]
    k = len(phis)
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
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def ApproxEntropy(U, m, r) -> float:
    """Approximate_entropy
    m represents the length of each compared run of data (essentially a window)
    r specifies a filtering level.
    """
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))



def generate_level_profile(levels,tmax,max_duration):
    t = 0
    array = np.array([])
    while (t < tmax):
        new_level = random.choice(levels)
        pulse_length = min((tmax - t), int(max_duration * random.random()))
        array = np.append(array, pulse_length * [new_level] )
        print("new level {} pulse {} array {}".format(new_level, pulse_length, array))
        t += pulse_length
        
    return array

def generate_random_profile(levels,tmax,max_duration):
    array = [random.choice(levels) for x in range(tmax)]
        
    return array



a = generate_level_profile([1,1,1,1,1], 100, 50)
b = generate_random_profile([-2,-1,0,1,2], 100, 30)
entropy_a = ApproxEntropy(a,1,0)
entropy_b = ApproxEntropy(b,1,0)

randU = np.random.choice([85, 80, 89], size=17*3)
print("test", ApproxEntropy(randU, 2, 3))

print("approximate entropies (leveled):{} (random):{}".format(entropy_a, entropy_b))

plt.plot(range(len(a)), a, 'r')
plt.plot(range(len(b)), b, 'b')
plt.legend(["leveled profile", "random profile"])
plt.title("Different profiles")
plt.show()
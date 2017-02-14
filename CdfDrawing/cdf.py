
import numpy as np
import matplotlib.pyplot as plt


def draw_cdf():
    data = np.loadtxt('err.txt')
    x = np.sort(data)
    y = np.arange(len(x)) / float(len(x)-1)
    plt.plot(x, y, label='some text')
    plt.grid()
    plt.legend()
    plt.xlabel('estimated error')
    plt.savefig('cdf.png')

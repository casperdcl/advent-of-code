import numpy as np

from ..sol2020 import conv


def day1():
    """Number of increments."""
    x = np.loadtxt("1.txt", dtype=np.int16)
    res1 = sum(x[1:] > x[:-1])
    y = conv(x, np.ones((3,)))
    res2 = sum(y[2:-1] > y[1:-2])

    return res1, res2

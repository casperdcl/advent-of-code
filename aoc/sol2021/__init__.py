import numpy as np

from ..sol2020 import conv


def day1():
    """Number of increments."""
    x = np.loadtxt("1.txt", dtype=np.int16)
    res1 = sum(x[1:] > x[:-1])
    y = conv(x, np.ones((3,)))
    res2 = sum(y[2:-1] > y[1:-2])

    return res1, res2


def day2():
    """2D navigation."""
    directions = {"forward": 1, "up": -1j, "down": 1j}
    with open("2.txt") as fd:
        x = [int(val) * directions[d] for i in fd for d, val in [i.split()]]
    res1 = sum(x)
    res1 = int(res1.real * res1.imag)

    aim = 0j
    res2 = complex()
    with open("2.txt") as fd:
        for i in fd:
            d, val = i.split()
            if d in {"up", "down"}:
                aim += {"up": -1j, "down": 1j}[d] * int(val)
            else:
                res2 += (directions[d] + aim) * int(val)
    res2 = int(res2.real * res2.imag)

    return res1, res2

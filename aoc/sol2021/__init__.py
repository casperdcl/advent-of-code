from collections import Counter

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


def day3():
    """Most common bits."""
    x = np.asarray(
        [list(map(int, num.strip())) for num in open("3.txt")], dtype=np.bool
    )

    def most_common(arr):
        """`max(Counter(arr).most_common())[0]` with tie-break using values"""
        return max(((num, val) for val, num in Counter(arr).most_common()))[1]

    most = np.array([most_common(i) for i in x.T])

    def binlist2int(arr):
        return sum(2 ** i * v for i, v in enumerate(reversed(arr)))

    res1 = binlist2int(most) * binlist2int(~most)

    o2, co2 = x, x
    for i in range(x.shape[1]):
        if len(o2) > 1:
            o2 = o2[o2[:, i] == most_common(o2[:, i])]
        if len(co2) > 1:
            co2 = co2[co2[:, i] != most_common(co2[:, i])]

    res2 = np.prod(list(map(binlist2int, [o2[0], co2[0]])))

    return res1, res2

    return res1, res2

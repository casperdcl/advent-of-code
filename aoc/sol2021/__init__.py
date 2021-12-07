import re
from collections import Counter

import numpy as np

from aoc.sol2020 import conv


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


def day4():
    """Winning & losing Bingo."""
    draws, boards = open("4.txt").read().strip().split("\n\n", 1)
    draws = [int(i) for i in draws.split(",")]
    boards = np.array(
        [[j.split() for j in i.split("\n")] for i in boards.split("\n\n")],
        dtype=np.int32,
    )
    marked = np.zeros(boards.shape, bool)

    for d in draws:
        marked[boards == d] = 1
        wins = marked.all(axis=1).any(axis=1) | marked.all(axis=2).any(axis=1)
        if any(wins):
            res1 = boards[wins][~marked[wins]].sum() * d
            break

    marked[:] = 0
    for d in draws:
        marked[boards == d] = 1
        wins = marked.all(axis=1).any(axis=1) | marked.all(axis=2).any(axis=1)
        if all(wins):
            marked[boards == d] = 0  # undo
            last = ~(marked.all(axis=1).any(axis=1) | marked.all(axis=2).any(axis=1))
            marked[boards == d] = 1  # redo

            res2 = boards[last][~marked[last]].sum() * d
            break

    return res1, res2


def day5():
    """Counting line intersections."""
    crds = np.array(
        [list(map(int, re.split(r"[^\d]+", i.strip()))) for i in open("5.txt")],
        dtype=np.int32,
    )
    grid = np.zeros((crds[:, ::2].max() + 1, crds[:, 1::2].max() + 1), dtype=np.int32)
    for x0, y0, x1, y1 in crds:
        if x0 == x1 or y0 == y1:  # horiz/vert
            grid[min(y0, y1) : max(y0, y1) + 1, min(x0, x1) : max(x0, x1) + 1] += 1
    res1 = (grid > 1).sum()

    for x0, y0, x1, y1 in crds:
        if abs(x1 - x0) == abs(y1 - y0):  # diag
            yd = 1 if y1 >= y0 else -1
            xd = 1 if x1 >= x0 else -1
            for y, x in zip(range(y0, y1 + yd, yd), range(x0, x1 + xd, xd)):
                grid[y, x] += 1
    res2 = (grid > 1).sum()

    return res1, res2


def day6():
    """Exponential population growth."""
    x = Counter(np.loadtxt("6.txt", delimiter=",", dtype=np.int8))

    for i in range(256):
        x = {k - 1: v for k, v in x.items()}
        if -1 in x:
            x[8] = x[-1]
            x.setdefault(6, 0)
            x[6] += x.pop(-1)
        if i == 79:
            res1 = sum(x.values())
    res2 = sum(x.values())

    return res1, res2


def day7():
    """Minimum total cost."""
    x = np.loadtxt("7.txt", delimiter=",", dtype=np.int16)

    res1 = min(sum(abs(x - i)) for i in range(max(x) + 1))

    costs = np.zeros((max(x) + 1,), dtype=np.uint32)
    costs[1] = 1
    for i in range(2, len(costs)):
        costs[i] = costs[i - 1] + i
    res2 = min(sum(costs[abs(x - i)]) for i in range(max(x) + 1))

    return res1, res2

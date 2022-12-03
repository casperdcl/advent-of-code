from heapq import nlargest

import numpy as np


def day1():
    """Reducing & sorting lists."""
    x = open("1.txt").read().strip().split("\n\n")
    x = nlargest(3, (sum(map(int, i.split())) for i in x))
    res1 = x[0]
    res2 = sum(x)
    return res1, res2


def day2():
    """Rock, paper, scissors."""
    games = np.loadtxt("2.txt", dtype="U1")
    trans = {"A": 0, "B": 1, "C": 2, "X": 0, "Y": 1, "Z": 2}
    res1, res2 = 0, 0
    for them, us in games:
        them = trans[them]

        us1 = trans[us]
        res1 += us1 + 1
        if us1 == them:  # draw
            res1 += 3
        elif (us1 - them) % 3 == 1:  # win
            res1 += 6

        us2 = {"X": (them - 1) % 3, "Y": them, "Z": (them + 1) % 3}[us]
        res2 += us2 + 1
        if us2 == them:  # draw
            res2 += 3
        elif (us2 - them) % 3 == 1:  # win
            res2 += 6

    return res1, res2


def day3():
    """Set intersections."""
    x = open("3.txt").read().strip().split()
    res1 = 0
    for i in x:
        N = len(i)
        c = set(i[: N // 2]).intersection(i[N // 2 :]).pop()
        res1 += ord(c) - (ord("a") - 1 if c.lower() == c else ord("A") - 27)
    res2, x2 = 0, iter(x)
    for i, j, k in zip(x2, x2, x2):  # zip(x[::3], x[1::3], x[2::3])
        c = set(i).intersection(j).intersection(k).pop()
        res2 += ord(c) - (ord("a") - 1 if c.lower() == c else ord("A") - 27)
    return res1, res2

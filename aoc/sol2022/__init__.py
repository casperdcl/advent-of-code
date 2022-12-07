from collections import deque
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


def day4():
    """Overlapping ranges."""
    abxy = np.fromregex(
        open("4.txt"), r"(\d+)-(\d+),(\d+)-(\d+)", [(c, np.int16) for c in "abxy"]
    )
    a, b, x, y = (abxy[c] for c in "abxy")
    res1 = sum(((a <= x) & (y <= b)) | ((x <= a) & (b <= y)))  # subset
    res2 = sum((x <= b) & (a <= y))  # overlap
    return res1, res2


def day5():
    """Stack manipulation."""
    yx, moves = open("5.txt").read().strip().split("\n\n")
    moves = [tuple(map(int, m.split()[1::2])) for m in moves.split("\n")]
    yx, N = yx.rsplit("\n", 1)
    N = int(N[-1][-1])
    yx = np.array([list(y[1::4].ljust(N)) for y in yx.split("\n")]).T[:, ::-1]

    stacks = [deque(i for i in stack if i != " ") for stack in yx]
    for n, src, dst in moves:
        stacks[dst - 1].extend(stacks[src - 1].pop() for _ in range(n))
    res1 = "".join(i.pop() for i in stacks)

    stacks = [deque(i for i in stack if i != " ") for stack in yx]
    for n, src, dst in moves:
        stacks[dst - 1].extend([stacks[src - 1].pop() for _ in range(n)][::-1])
    res2 = "".join(i.pop() for i in stacks)
    return res1, res2


def day6():
    """Unique sequences."""
    x = open("6.txt").read().strip()
    res1 = None
    buf1 = deque([], maxlen=4)
    buf2 = deque([], maxlen=14)
    for i, c in enumerate(x):
        if not res1:
            buf1.append(c)
            # TODO: use `set.{add,remove}` O(1) instead of re-creating sets O(N)
            if len(set(buf1)) == 4:
                res1 = i + 1
        buf2.append(c)
        if len(set(buf2)) == 14:
            res2 = i + 1
            break
    return res1, res2

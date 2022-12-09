from collections import deque
from functools import cache
from heapq import nlargest
from itertools import pairwise

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

    stacks = [None] + [deque(i for i in stack if i != " ") for stack in yx]
    for n, src, dst in moves:
        stacks[dst].extend(stacks[src].pop() for _ in range(n))
    res1 = "".join(i.pop() for i in stacks[1:])

    stacks = [None] + [deque(i for i in stack if i != " ") for stack in yx]
    for n, src, dst in moves:
        stacks[dst].extend([stacks[src].pop() for _ in range(n)][::-1])
    res2 = "".join(i.pop() for i in stacks[1:])
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
            if len(set(buf1)) == buf1.maxlen:
                res1 = i + 1
        buf2.append(c)
        if len(set(buf2)) == buf2.maxlen:
            res2 = i + 1
            break
    return res1, res2


class MultiNode:
    def __init__(self, name, value=0, parent=None):
        self.name = name
        self.value = value
        self.children = set()
        self.parent = parent
        if parent is not None:
            parent.children.add(self)

    @cache  # assumes no further modifications to `self.children`
    def __len__(self):
        return sum(map(len, self.children)) + self.value

    def __getitem__(self, name):
        return next(filter(lambda i: i.name == name, self.children))

    def __iter__(self):
        yield self
        for i in self.children:
            if i.children:  # exclude leaves
                yield from i

    def add(self, *args, **kwargs):
        return MultiNode(*args, parent=self, **kwargs)


def day7():
    """Directory sizes."""
    tty = open("7.txt").read().strip().split("\n")
    root = cwd = MultiNode("/")
    for l in tty:
        match l.split():
            case "$", "ls":
                pass
            case "$", "cd", "..":
                cwd = cwd.parent
            case "$", "cd", "/":
                cwd = root
            case "$", "cd", subdir:
                cwd = cwd[subdir]  # except: cwd = cwd.add(subdir)
            case "dir", subdir:
                cwd.add(subdir)
            case size, filename:
                cwd.add(filename, value=int(size))
            case _:
                raise ValueError(l)

    res1 = sum(len(cwd) for cwd in root if len(cwd) <= 100_000)
    diff = 30_000_000 - (70_000_000 - len(root))
    res2 = min(len(cwd) for cwd in root if len(cwd) >= diff)
    return res1, res2


def day8():
    """Max block reduce."""
    grid = np.genfromtxt("8.txt", dtype=np.int8, delimiter=1)
    vis = np.zeros_like(grid, dtype=bool)
    for _ in range(4):
        vis_north = np.zeros_like(grid) - 1
        for y0, y in pairwise(range(grid.shape[0])):
            vis_north[y] = np.max((vis_north[y0], grid[y0]), axis=0)
        vis |= grid > vis_north
        grid, vis = map(np.rot90, (grid, vis))
    res1 = vis.sum()

    vis = np.ones_like(grid, dtype=np.int32)
    for _ in range(4):
        for y in range(grid.shape[0]):
            vis_north = np.zeros_like(grid[y])
            done = np.zeros_like(grid[y], dtype=bool)
            for yy in range(y - 1, -1, -1):
                vis_north[~done] += 1
                done[grid[yy] >= grid[y]] = True
                if done.all():
                    break
            vis[y] *= vis_north
        grid, vis = map(np.rot90, (grid, vis))
    res2 = vis.max()
    return res1, res2


def day9():
    """Chasing 2D points."""
    knots = np.zeros((10, 2), dtype=np.int16)
    visited = {1: {(0, 0)}, 9: {(0, 0)}}  # {knot_index: {(y, x), ...}}
    for d, steps in np.loadtxt("9.txt", dtype=[("d", "U1"), ("steps", np.int8)]):
        for _ in range(steps):
            knots[0][0 if d in "UD" else 1] += 1 if d in "RD" else -1  # head
            for head, tail in pairwise(range(len(knots))):
                if (np.abs(diff := knots[head] - knots[tail]) > 1).any():
                    # more than 1 space away
                    knots[tail] += np.clip(diff, -1, 1)  # move 1 step closer
                    if tail in visited:
                        visited[tail].add(tuple(knots[tail]))
                else:
                    break
    res1, res2 = map(len, visited.values())
    return res1, res2

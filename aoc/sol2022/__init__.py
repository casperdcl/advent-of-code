"""Solutions to https://adventofcode.com/2022"""
import re
import string
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from heapq import nlargest
from itertools import pairwise
from math import lcm
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np

from aoc.sol2020 import conv
from aoc.sol2021 import plot_binary


def day1():
    """Reducing & sorting lists."""
    x = open("1.txt").read().strip().split("\n\n")
    x = nlargest(3, (sum(map(int, i.split())) for i in x))
    return x[0], sum(x)


def day2():
    """Rock, paper, scissors."""
    games = np.loadtxt("2.txt", dtype="U1")
    base_score = {"A": 0, "B": 1, "C": 2, "X": 0, "Y": 1, "Z": 2}
    draw_win_lose_score = 4, 7, 1
    res1, res2 = 0, 0
    for them, us in games:
        them, us1 = base_score[them], base_score[us]
        res1 += us1 + draw_win_lose_score[(us1 - them) % 3]
        us2 = (them + us1 - 1) % 3  # (them + {"X": -1, "Y": 0, "Z": 1}[us]) % 3
        res2 += us2 + draw_win_lose_score[(us2 - them) % 3]
    return res1, res2


def day3():
    """Set intersections."""
    x = open("3.txt").read().strip().split()
    priorities = {c: i for i, c in enumerate(string.ascii_letters, 1)}
    res1 = sum(
        priorities[set(i[: (half := len(i) // 2)]).intersection(i[half:]).pop()]
        for i in x
    )
    x2 = iter(x)
    res2 = sum(
        priorities[set(i).intersection(j, k).pop()]
        for i, j, k in zip(x2, x2, x2)  # zip(x[::3], x[1::3], x[2::3])
    )
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

    def solve(step):
        stacks = [None] + [[i for i in stack if i != " "] for stack in yx]
        for n, src, dst in moves:
            stacks[dst].extend(stacks[src][-n:][::step])
            stacks[src] = stacks[src][:-n]
        return "".join(i[-1] for i in stacks[1:])

    return solve(-1), solve(1)


def day6():
    """Unique sequences."""
    x = open("6.txt").read().strip()
    res = []
    for maxlen in (4, 14):
        buf = deque([], maxlen=maxlen)
        for i, c in enumerate(x):
            buf.append(c)
            if len(set(buf)) == maxlen:
                res.append(i + 1)
                break
    return tuple(res)


def day7():
    """Directory sizes."""
    tty = open("7.txt").read().strip().split("\n")
    sizes = defaultdict(int)  # {path: size}
    cwd = Path("/")
    for l in tty:
        match l.split():
            case ("$", "ls") | ("dir", _):
                pass
            case "$", "cd", "..":
                cwd = cwd.parent
            case "$", "cd", "/":
                cwd = Path("/")
            case "$", "cd", subdir:
                cwd /= subdir
            case size, _:  # increase size of all directories in the tree
                for path in (cwd / "_").parents:
                    sizes[path] += int(size)
            case _:
                raise ValueError(l)

    res1 = sum(filter(lambda s: s <= 100_000, sizes.values()))
    diff = 30_000_000 - (70_000_000 - sizes[Path("/")])
    res2 = min(filter(lambda s: s >= diff, sizes.values()))
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


def day10():
    """CRT drawing in assembly."""
    x, cycle, res1 = 1, 0, 0
    res2 = np.empty((6, 40), dtype=bool)

    def step(x, cycle, strength):
        Y, X = divmod(cycle, 40)  # CRT position
        res2[Y, X] = -2 < X - x < 2  # overlaps sprite position
        cycle += 1
        return cycle, strength + (cycle * x if cycle % 40 == 20 else 0)

    for pc in open("10.txt").read().strip().split("\n"):
        match pc.split():
            case "addx", v:
                for _ in range(2):
                    cycle, res1 = step(x, cycle, res1)
                x += int(v)
            case ("noop",):
                cycle, res1 = step(x, cycle, res1)
            case _:
                raise ValueError(pc)

    return res1, plot_binary(res2)


def day11():
    """Passing parcels."""

    @dataclass(slots=True)
    class Monkey:
        items: list[int]
        op: Callable[[int], int]
        div: int
        y: int
        n: int
        seen: int = 0

    mnks = tuple(
        Monkey(
            list(map(int, items.split(", "))),
            eval(f"lambda old: {op}"),
            *map(int, (div, y, n)),
        )
        for items, op, div, y, n in re.findall(
            r"""Monkey \d+:
  Starting items: ([0-9, ]*)
  Operation: new = (.*)
  Test: divisible by (\d+)
    If true: throw to monkey (\d+)
    If false: throw to monkey (\d+)""",
            open("11.txt").read(),
            flags=re.M,
        )
    )
    LCM = lcm(*{m.div for m in mnks})

    def solve(mnks, rounds, divisor):
        for _ in range(rounds):
            for m in mnks:
                for old in m.items:
                    new = (m.op(old) // divisor) % LCM
                    mnks[m.n if new % m.div else m.y].items.append(new)
                m.seen += len(m.items)
                m.items.clear()
        return np.prod(nlargest(2, (m.seen for m in mnks)))

    return solve(deepcopy(mnks), 20, 3), solve(mnks, 10_000, 1)


def day12():
    """Shortest path."""
    grid = np.array(
        [list(map(ord, y)) for y in open("12.txt").read().strip().split()],
        dtype=np.int8,
    ) - ord("a")
    [yS], [xS] = np.where(grid == ord("S") - ord("a"))  # start
    [yE], [xE] = np.where(grid == ord("E") - ord("a"))  # end
    grid[[yS, yE], [xS, xE]] = vmin, vmax = 0, 25

    g = nx.DiGraph()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):  # north south west east
        knl = np.zeros((3, 3), dtype=np.int8)
        knl[1, 1], knl[1 + dy, 1 + dx] = -1, 1
        for y, x in zip(
            *np.where(conv(grid, knl[::-1, ::-1], mode="constant", cval=vmax + 2) < 2)
        ):
            g.add_edge((y, x), (y + dy, x + dx))
    res1 = nx.shortest_path_length(g, (yS, xS), (yE, xE))

    res2 = grid.size
    for yS, xS in zip(*np.where(grid == vmin)):
        try:
            res2 = min(res2, nx.shortest_path_length(g, (yS, xS), (yE, xE)))
        except nx.NetworkXNoPath:
            pass
    return res1, res2

import re
from collections import Counter
from functools import reduce
from itertools import permutations, product

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
        x = [int((d_val := i.split())[1]) * directions[d_val[0]] for i in fd]
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
    marked = np.zeros(boards.shape, dtype=np.bool)

    res1 = None
    for d in draws:
        marked[boards == d] = 1
        wins = marked.all(axis=1).any(axis=1) | marked.all(axis=2).any(axis=1)
        if res1 is None and any(wins):
            res1 = boards[wins][~marked[wins]].sum() * d
        elif all(wins):
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

    nums = [0] * 9
    for i, num in x.items():
        nums[i] = num
    for day in range(256):
        spawn = nums[0]
        for i in range(8):
            nums[i] = nums[i + 1]
        nums[8] = spawn
        nums[6] += spawn
        if day == 79:
            res1 = sum(nums)
    res2 = sum(nums)

    return res1, res2


def day7(brute=False):
    """Minimum total cost."""
    x = np.loadtxt("7.txt", delimiter=",", dtype=np.int16)

    if brute:
        res1 = min(sum(abs(x - i)) for i in range(max(x) + 1))
    res1 = sum(abs(x - np.median(x).astype(np.int16)))

    if brute:
        res2 = min(
            sum((costs := abs(x - i).astype(np.int32)) * (costs + 1) // 2)
            for i in range(max(x) + 1)
        )
    costs = abs(x - x.mean().astype(np.int16)).astype(np.int32)
    res2 = sum(costs * (costs + 1) // 2)

    return res1, res2


def day8(brute=False):
    """Deducing 7-segment displays."""

    def sort_str(s):
        return "".join(sorted(s))

    x = [
        [[sort_str(k) for k in j.strip().split()] for j in i.split(" | ")]
        for i in open("8.txt")
    ]

    # Brute force translation table solution
    if brute:
        maps = {
            "abcefg": 0,
            "cf": 1,
            "acdeg": 2,
            "acdfg": 3,
            "bcdf": 4,
            "abdfg": 5,
            "abdefg": 6,
            "acf": 7,
            "abcdefg": 8,
            "abcdfg": 9,
        }
        res1, res2 = 0, np.zeros((4,), dtype=np.int32)
        for src, out in x:
            res1 += sum(len(i) in {2, 3, 4, 7} for i in out)
            for perm in permutations("abcdefg"):
                tab = str.maketrans("abcdefg", "".join(perm))
                src_trans = (sort_str(i.translate(tab)) for i in src)
                if all(i in maps for i in src_trans):
                    out_trans = (sort_str(i.translate(tab)) for i in out)
                    res2 += [maps[i] for i in out_trans]
                    break
        res2 = sum(10 ** i * v for i, v in enumerate(res2[::-1]))

        return res1, res2

    # 100x faster manual solution
    segs = {2: 1, 4: 4, 3: 7, 7: 8}
    res1 = sum(len([len(i) for i in out if len(i) in segs]) for _, out in x)

    seg_appearances = {
        9: {0, 1, 3, 4, 5, 6, 7, 8, 9},
        8: {0, 2, 3, 5, 6, 7, 8, 9} | {0, 1, 2, 3, 4, 7, 8, 9},
        7: {2, 3, 4, 5, 6, 8, 9} | {0, 2, 3, 5, 6, 8, 9},
        6: {0, 4, 5, 6, 8, 9},
        4: {0, 2, 6, 8},
    }
    res2 = np.zeros((4,), dtype=np.int32)
    for src, out in x:
        maps = {}
        # 1, 4, 7, 8
        for i in src:
            if len(i) in segs:
                maps[i] = {segs[len(i)]}

        # 2, 3
        for seg, num in Counter("".join(src)).items():
            for i in src:
                if seg in i:
                    maps.setdefault(i, seg_appearances[num].copy())
                    maps[i] &= seg_appearances[num]
        sols = {list(i)[0] for i in maps.values() if len(i) == 1}
        new_sols = {
            val
            for val, num in Counter(
                j for i in maps.values() if len(i) > 1 for j in i
            ).items()
            if num == 1
        }
        for i in maps:
            if maps[i] & new_sols:
                maps[i] &= new_sols
            elif len(maps[i]) > 1 and maps[i] & sols:
                maps[i] -= sols

        # 0, 5, 6, 9
        seven = set(next(iter(k for k, c in maps.items() if c == {7})))
        for k, candidates in maps.items():
            if candidates == {0, 6}:
                if len(set(k) & seven) == 3:
                    candidates &= {0}
                else:
                    candidates &= {6}
            elif candidates & {5, 9}:
                if len(set(k) & seven) == 3:
                    candidates &= {9}
                else:
                    candidates &= {5}

        assert all(len(i) == 1 for i in maps.values())
        maps = {k: v.pop() for k, v in maps.items()}
        res2 += [maps[disp] for disp in out]
    res2 = sum(10 ** i * v for i, v in enumerate(res2[::-1]))

    return res1, res2


def day9():
    """2D segmentation."""
    x = np.array(
        [list(map(int, i)) for i in open("9.txt").read().strip().split()], dtype=np.int8
    )

    adj = np.zeros((4, 3, 3), dtype=x.dtype)
    adj[0, 1, 0] = adj[1, 0, 1] = adj[2, 1, 2] = adj[3, 2, 1] = 1
    lows = np.all([x < conv(x, a, cval=10) for a in adj], axis=0)
    res1 = (x[lows] + 1).sum()

    borders = x >= 9
    visited = np.zeros_like(borders)

    def basin_size(j, i):
        if (
            j < 0
            or i < 0
            or j >= visited.shape[0]
            or i >= visited.shape[1]
            or visited[j, i]
            or borders[j, i]
        ):
            return 0
        visited[j, i] = 1
        return 1 + sum(
            basin_size(j + b, i + a)
            for b, a in product(range(-1, 2), repeat=2)
            if abs(a) != abs(b)
        )

    res2 = np.prod(
        sorted(
            basin_size(j, i)
            for j, i in product(range(x.shape[0]), range(x.shape[1]))
            if lows[j, i]
        )[-3:]
    )

    return res1, res2


def day10():
    """Brace matching errors."""
    x = open("10.txt").read().strip().split()
    close = {")": "(", "]": "[", "}": "{", ">": "<"}
    score_err = {")": 3, "]": 57, "}": 1197, ">": 25137}
    score_end = {"(": 1, "[": 2, "{": 3, "<": 4}

    res1, res2 = 0, []
    for line in x:
        last = []
        for c in line:
            if c in "([{<":
                last.append(c)
            elif close[c] != last.pop():
                res1 += score_err[c]  # syntax error
                break
        else:  # missing closures
            res2.append(reduce(lambda i, c: 5 * i + score_end[c], last[::-1], 0))

    return res1, int(np.median(res2))

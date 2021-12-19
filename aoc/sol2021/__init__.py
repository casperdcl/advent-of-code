import operator
import re
from collections import Counter
from functools import lru_cache, reduce
from io import StringIO
from itertools import count, permutations, product

import networkx as nx
import numpy as np
from tqdm import tqdm, trange

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
    x = [int((d_val := i.split())[1]) * directions[d_val[0]] for i in open("2.txt")]
    res1 = sum(x)
    res1 = int(res1.real * res1.imag)

    aim = 0j
    res2 = 0j
    for i in open("2.txt"):
        d, val = i.split()
        if d in {"up", "down"}:
            aim += {"up": -1j, "down": 1j}[d] * int(val)
        else:
            res2 += (directions[d] + aim) * int(val)
    res2 = int(res2.real * res2.imag)

    return res1, res2


def day3():
    """Most common bits."""
    x = np.asarray([list(map(int, num.strip())) for num in open("3.txt")], dtype=bool)

    def most_common(arr):
        """`max(Counter(arr).most_common())[0]` with tie-break using values"""
        return max(((num, val) for val, num in Counter(arr).most_common()))[1]

    most = np.array([most_common(i) for i in x.T])

    def binlist2int(arr):
        return sum(2 ** i * v for i, v in enumerate(arr[::-1]))

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
    draws = np.fromstring(draws, dtype=np.int8, sep=",")
    boards = np.array(
        [[j.split() for j in i.split("\n")] for i in boards.split("\n\n")],
        dtype=np.int8,
    )
    marked = np.zeros(boards.shape, dtype=bool)

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
    crds = np.fromregex(
        "5.txt",
        r"(\d+),(\d+) -> (\d+),(\d+)",
        [("x0", np.int16), ("y0", np.int16), ("x1", np.int16), ("y1", np.int16)],
    )
    grid = np.zeros(
        (max(map(max, crds[["x0", "x1"]])) + 1, max(map(max, crds[["y0", "y1"]])) + 1),
        dtype=np.int16,
    )
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

    return res1, sum(nums)


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


def day11():
    """Game of flash."""
    x = np.asarray(
        [list(map(int, i)) for i in open("11.txt").read().strip().split()],
        dtype=np.uint8,
    )
    adj = np.ones((3, 3), dtype=x.dtype)
    adj[1, 1] = 0

    res1 = 0
    for step in count(1):
        x += 1
        while True:
            cur = x == 10  # flashing
            new = conv(cur.astype(x.dtype), adj)
            new[x > 9] = 0
            if not new.any():  # no further energy increases
                break
            x[cur] = 11  # max energy
            x[msk] = np.clip(x[msk := new.astype(bool)] + new[msk], 0, 10)
        cur = x > 9  # flashed
        x[cur] = 0  # reset max energy
        if step <= 100:
            res1 += cur.sum()
        if cur.all():
            return res1, step


def day12():
    """Graph 2nd order paths."""
    g = nx.Graph()
    for i in open("12.txt"):
        g.add_edge(*i.strip().split("-"))

    def is_upper(s):
        return s.upper() == s

    def recurse(node="start", visited=None, allow_twice=False):
        if node == "end":
            return 1
        visited = set() if visited is None else visited.copy()
        if not is_upper(node):
            visited |= {node}
        res = 0
        for n in g[node]:
            if n != "start":
                if n not in visited:
                    res += recurse(n, visited, allow_twice)
                elif allow_twice:
                    res += recurse(n, visited)
        return res

    return recurse(), recurse(allow_twice=True)


def day13():
    """Folding paper."""
    xy, folds = open("13.txt").read().strip().split("\n\n", 1)
    xy = np.loadtxt(StringIO(xy), delimiter=",", dtype=np.int16)
    folds = np.fromregex(
        StringIO(folds), r"fold along ([xy])=(\d+)", [("ax", "U1"), ("i", np.int16)]
    )
    grid = np.zeros(xy.max(axis=0)[::-1] + 1, dtype=bool)
    grid[xy[:, 1], xy[:, 0]] = True

    res1 = None
    for ax, i in folds:
        if ax == "x":
            f = grid[:, i + 1 :][:, ::-1]
            grid = grid[:, :i] | np.pad(f, ((0, 0), (grid.shape[0] - f.shape[0], 0)))
        else:
            f = grid[i + 1 :][::-1]
            grid = grid[:i] | np.pad(f, ((grid.shape[1] - f.shape[1], 0), (0, 0)))
        if res1 is None:
            res1 = grid.sum()

    print("\n".join("".join("\u2588" if x else " " for x in y) for y in grid))

    return res1, f"^plot ({grid.sum()})^"


def day14():
    """Depth First Counter."""
    tmp, pairs = open("14.txt").read().strip().split("\n\n", 1)
    pairs = dict(i.split(" -> ") for i in pairs.split("\n"))

    @lru_cache(maxsize=5000)
    def recurse(l, r, step, maxstep):
        if step == maxstep:
            return Counter()
        return (
            Counter(new := pairs[l + r])
            + recurse(l, new, step + 1, maxstep)
            + recurse(new, r, step + 1, maxstep)
        )

    c1, c2 = Counter(tmp), Counter(tmp)
    for i in range(len(tmp) - 1):
        c1 += recurse(tmp[i], tmp[i + 1], 0, 10)
        c2 += recurse(tmp[i], tmp[i + 1], 0, 40)

    return max(c1.values()) - min(c1.values()), max(c2.values()) - min(c2.values())


def day15():
    """Shortest path."""
    w = np.asarray(
        [list(map(int, i)) for i in open("15.txt").read().strip().split()],
        dtype=np.uint8,
    )
    shp = w.shape
    for ax in range(2):
        w = np.concatenate([(w + i) % 9 + 1 for i in range(-1, 4)], axis=ax)
    g = nx.DiGraph()

    def add_edges(w, progress=False):
        neighbours = [
            (b, a) for b, a in product(range(-1, 2), repeat=2) if abs(a) != abs(b)
        ]
        for j, i in tqdm(
            product(range(w.shape[0]), range(w.shape[1])),
            unit="node",
            disable=not progress,
            leave=False,
            total=np.prod(w.shape),
            unit_scale=True,
        ):
            for b, a in neighbours:
                if 0 <= j + b < w.shape[0] and 0 <= i + a < w.shape[1]:
                    g.add_edge((j, i), (j + b, i + a), w=w[j + b][i + a])

    add_edges(w[: shp[0], : shp[1]])
    res1 = nx.shortest_path_length(g, (0, 0), tuple(i - 1 for i in shp), weight="w")
    add_edges(w, True)
    res2 = nx.shortest_path_length(g, (0, 0), tuple(i - 1 for i in w.shape), weight="w")

    return res1, res2


def day16():
    """Parsing machine code."""
    x = "".join(f"{bin(int(i, 16))[2:]:>04}" for i in open("16.txt").read().strip())
    op = {0: "sm", 1: "pd", 2: "mn", 3: "mx", 5: "gt", 6: "lt", 7: "eq"}
    ops = {
        "sm": lambda *i: sum(i),
        "pd": lambda *i: np.prod(i),
        "mn": lambda *i: min(i),
        "mx": lambda *i: max(i),
        "gt": operator.gt,
        "lt": operator.lt,
        "eq": operator.eq,
    }
    res1, res2 = [], []

    def parse(x: str, i: int, X: int, once: bool = False):
        while X - i >= 8 or not all(j == "0" for j in x[i:X]):
            ver = int(x[i : (i := i + 3)], 2)
            res1.append(ver)
            typ = int(x[i : (i := i + 3)], 2)
            if typ == 4:  # literal
                delta = (x[i::5].index("0") + 1) * 5
                res2.append(
                    "0b" + "".join(re.findall(".(.{4})", x[i : (i := i + delta)]))
                )
            else:  # operator
                res2.append(f"{op[typ]}(")
                i += 1
                if x[i - 1] == "0":
                    sub_len = int(x[i : (i := i + 15)], 2)
                    parse(x, i, (i := i + sub_len))
                else:
                    for _ in range(int(x[i : (i := i + 11)], 2)):
                        i = parse(x, i, X, once=True)
                res2.append(")")
            if once:
                return i
        return X

    parse(x, 0, len(x))
    return sum(res1), eval(",".join(res2).replace("(,", "("), ops)


def crange(start: complex, stop: complex, step: complex = 1):
    while start != stop:
        yield start
        start += step


def day17():
    """Discrete projectile targets."""
    x0, x1, y0, y1 = np.fromregex(
        open("17.txt"),
        r"target area: x=([-\d]+)..([-\d]+), y=([-\d]+)..([-\d]+)",
        [("x0", np.int16), ("x1", np.int16), ("y0", np.int16), ("y1", np.int16)],
    )[0]

    res1, res2 = None, 0
    for u in trange(-y0, y0 - 1, -1, leave=False):
        for v in crange(1 + 1j * u, 1 + x1 + 1j * u):
            p, height = 0j, 0
            while p.real <= x1 and y0 <= p.imag:
                p += v
                height = max(height, int(p.imag))
                v -= (1 if v.real > 0 else 0) + 1j
                if x0 <= p.real <= x1 and y0 <= p.imag <= y1:
                    if res1 is None:
                        res1 = height
                    res2 += 1
                    break

    return res1, res2


def day18():
    """Binary Tree custom addition."""

    class BSTNode:
        def __init__(self, val=None, left=None, right=None, parent=None):
            assert val is None or (left is None and right is None)
            self.val, self.left, self.right, self.parent = val, left, right, parent
            for n in (left, right):
                if n:
                    n.parent = self

        def __repr__(self):
            return f"[{self.left},{self.right}]" if self.val is None else repr(self.val)

        def build(self, pairs, dtype=int):
            if isinstance(pairs, dtype):
                self.val = pairs
            else:
                self.left = BSTNode(parent=self).build(pairs[0], dtype)
                self.right = BSTNode(parent=self).build(pairs[1], dtype)
            return self

        def copy(self, parent=None):
            # return BSTNode().build(eval(repr(self)))
            res = BSTNode(val=self.val, parent=parent)
            for n in ("left", "right"):
                if o := getattr(self, n):
                    setattr(res, n, o.copy(res))
            return res

        def first(self):
            return self.left.first() if self.left else self

        def last(self):
            return self.right.last() if self.right else self

        def leaves(self):
            if self.left:
                yield from self.left.leaves()
            if self.val is not None:
                yield self
            if self.right:
                yield from self.right.leaves()

        def is_left_child(self):
            return getattr(self.parent, "left", None) is self

        def is_right_child(self):
            return getattr(self.parent, "right", None) is self

        def next_leaf(self):
            n = self
            while n.is_right_child():
                n = n.parent
            if n.is_left_child():
                return n.parent.right.first()
            return None

        def prev_leaf(self):
            n = self
            while n.is_left_child():
                n = n.parent
            if n.is_right_child():
                return n.parent.left.last()
            return None

        def num_parents(self):
            return 1 + self.parent.num_parents() if self.parent else 0

        def explode(self):
            for n in self.leaves():
                if n.num_parents() == 5:
                    o = n.next_leaf()
                    if m := n.prev_leaf():
                        m.val += n.val
                    if p := o.next_leaf():
                        p.val += o.val
                    n.parent.val, n.parent.left, n.parent.right = 0, None, None
                    return self

        def split(self):
            for n in self.leaves():
                if n.val >= 10:
                    res, rem = divmod(n.val, 2)
                    n.left = BSTNode(parent=n, val=res)
                    n.right = BSTNode(parent=n, val=res + rem)
                    n.val = None
                    return self

        def explode_split(self):
            while True:
                while self.explode():
                    pass
                if self.split() is None:
                    break
            return self

        def __add__(self, other):
            return BSTNode(left=self.copy(), right=other.copy()).explode_split()

        def __abs__(self):
            return sum(
                abs(n) * scale
                for n, scale in [(self.left, 3), (self.val, 1), (self.right, 2)]
                if n is not None
            )

    x = [BSTNode().build(eval(i)) for i in open("18.txt")]

    return abs(sum(x[1:], x[0])), max(
        abs(m + n)
        for m, n in tqdm(product(x, repeat=2), total=len(x) ** 2, leave=False)
    )

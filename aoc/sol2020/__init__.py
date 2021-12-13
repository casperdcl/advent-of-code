import re
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from functools import lru_cache, partial, reduce
from itertools import chain, combinations, count, islice, product

import networkx as nx
import numpy as np
import yaml
from scipy.ndimage import convolve
from tqdm import trange


def day1():
    """
    Pairs/triplets of numbers which sum to 2020.
    Returns product of answers.
    """
    x = np.loadtxt("1.txt", dtype=np.int16)
    set_x = set(x)

    y = set(2020 - x)
    res1 = np.prod(list(set_x & y))

    for target in y:
        z = set(target - x)
        res2 = set_x & z
        if res2:
            res2 = list(res2)
            res2.append(2020 - sum(res2))
            assert res2[-1] in x
            res2 = np.prod(res2)
            break

    return res1, res2


def day2():
    """Number of valid passwords."""
    x = open("2.txt").read()
    x = re.findall(r"^(\d+)-(\d+) (\w): (.+)$", x, flags=re.M)

    res1 = sum(
        1 if int(cmin) <= pwd.count(c) <= int(cmax) else 0 for cmin, cmax, c, pwd in x
    )

    res2 = sum(
        1 if (pwd[int(i) - 1] + pwd[int(j) - 1]).count(c) == 1 else 0
        for i, j, c, pwd in x
    )

    return res1, res2


def day3():
    """Number of #s along a diagonal path."""
    x = open("3.txt").read().strip()
    x = [[1 if i == "#" else 0 for i in row] for row in x.split("\n")]

    cols = len(x[0])
    res1 = sum(row[(3 * r) % cols] for r, row in enumerate(x))

    res2 = 1
    for cstride in [1, 3, 5, 7]:
        res2 *= sum(row[(cstride * r) % cols] for r, row in enumerate(x))
    for rstride in [2]:
        res2 *= sum(
            row[(r // rstride) % cols] for r, row in enumerate(x) if r % rstride == 0
        )

    return res1, res2


def day4():
    """Number of valid passports."""
    x = open("4.txt").read().strip()
    x = [dict(i.split(":") for i in row.split()) for row in x.split("\n\n")]
    REQUIRED = {"byr", "iyr", "eyr", "hgt", "hcl", "ecl", "pid"}  # "cid"
    x = list(filter(lambda i: not (REQUIRED - i.keys()), x))
    res1 = len(x)

    x = list(
        filter(
            lambda i: 1920 <= int(i["byr"]) <= 2002
            and 2010 <= int(i["iyr"]) <= 2020
            and 2020 <= int(i["eyr"]) <= 2030
            and (
                (i["hgt"][-2:] == "cm" and 150 <= int(i["hgt"][:-2]) <= 193)
                or (i["hgt"][-2:] == "in" and 59 <= int(i["hgt"][:-2]) <= 76)
            )
            and re.match(r"^#[0-9a-f]{6}$", i["hcl"])
            and re.match(r"^(amb|blu|brn|gry|grn|hzl|oth)$", i["ecl"])
            and re.match(r"^\d{9}$", i["pid"]),
            x,
        )
    )
    res2 = len(x)

    return res1, res2


def day5():
    """2D binary space partitioning plane seat IDs."""
    x = open("5.txt").read().strip()
    for i, j in ["F0", "B1", "L0", "R1"]:
        x = x.replace(i, j)
    x = [int(row[:7], 2) * 8 + int(row[7:], 2) for row in x.split("\n")]
    res1 = max(x)

    x = np.sort(x)
    res2 = x[:-1][x[1:] - x[:-1] > 1][0] + 1  # missing ID

    return res1, res2


def intersection(sets):
    """Intersection of all given sets"""
    return reduce(lambda x, y: x & y, sets)


def day6():
    """Common choices."""
    x = open("6.txt").read().strip().split("\n\n")
    res1 = sum(len({i for row in batch.split() for i in row}) for batch in x)
    res2 = sum(len(intersection(set(row) for row in batch.split())) for batch in x)
    return res1, res2


def day7():
    """Counting matryoshka bags."""
    x = open("7.txt").read().strip().split("\n")

    g = nx.DiGraph()
    for i in x:
        i, outs = re.match("^(.*?) bags contain (.*).$", i).groups()
        outs = filter(None, re.findall("(?:[0-9]+ (.*?)|no other) bags?", outs))
        for j in outs:
            g.add_edge(j, i)
    res1 = len(nx.single_source_shortest_path(g, "shiny gold")) - 1

    g = nx.DiGraph()
    for i in x:
        i, outs = re.match("^(.*?) bags contain (.*).$", i).groups()
        outs = re.findall(r"\b([0-9]+|no) (.*?) bags?", outs)
        for k, j in outs:
            if (k, j) != ("no", "other"):
                g.add_edge(i, j, weight=int(k))

    @lru_cache(maxsize=len(g))
    def get_sum(i):
        return 1 + sum(get_sum(j) * k["weight"] for j, k in g[i].items())

    res2 = sum(get_sum(j) * k["weight"] for j, k in g["shiny gold"].items())

    return res1, res2


def day8():
    """Breaking infinite loops in Assembly."""
    x = open("8.txt").read()
    x = [
        (i, int(v)) for i, v in re.findall(r"^(nop|acc|jmp) ([+-]\d+)$", x, flags=re.M)
    ]

    def run(x, visited=None):
        visited = visited or set()
        pc = 0  # program counter
        acc = 0  # accumulator
        while True:
            if pc in visited:
                break
            visited.add(pc)
            if x[pc][0] == "jmp":
                pc += x[pc][1]
            else:
                if x[pc][0] == "acc":
                    acc += x[pc][1]
                pc += 1
        return acc

    res1 = run(x)

    g = nx.DiGraph()
    for pc, (i, v) in enumerate(x):
        g.add_edge(pc, pc + (v if i == "jmp" else 1))

    def loop_exists(g):
        return next(nx.simple_cycles(g), [])

    for pc in loop_exists(g):
        i = x[pc][0]
        pc2 = list(g[pc].keys())[0]
        if i in ("jmp", "nop"):
            g.remove_edge(pc, pc2)
            if i == "jmp":
                g.add_edge(pc, pc + 1)
                if not loop_exists(g):
                    break
                g.remove_edge(pc, pc + 1)
            else:  # i == "nop"
                g.add_edge(pc, pc + x[pc][1])
                if not loop_exists(g):
                    break
                g.remove_edge(pc, pc + x[pc][1])
            g.add_edge(pc, pc2, i=i)
    else:
        raise ValueError("could not break loop")

    x[pc] = ("jmp" if x[pc][0] == "nop" else "nop", x[pc][1])
    res2 = run(x, {len(x)})

    return res1, res2


def day9():
    """eXchange-Masking Addition System (XMAS)."""
    x = np.loadtxt("9.txt", dtype=np.int64)
    mask = 25

    for i in range(mask, len(x)):
        prev = x[i - mask : i]
        other = set(x[i] - prev) & set(prev)
        if not other or (len(other) == 1 and prev.count(x[i] / 2) > 1):
            res1 = x[i]
            break
    else:
        raise IndexError("no solution found")

    for i in range(len(x)):
        j = i + 2
        s = x[i:j].sum()
        while s < res1:
            s += x[j]
            j += 1
        if s == res1:
            res2 = x[i:j].min() + x[i:j].max()
            break
    else:
        raise IndexError("not found")

    return res1, res2


def day10():
    """Chain of adapters."""
    x = np.loadtxt("10.txt", dtype=np.int16)
    x = np.concatenate([[0], x, [x.max() + 3]])
    x.sort()
    diff = (x[1:] - x[:-1]).tolist()
    res1 = diff.count(3) * diff.count(1)

    assert set(diff) == {1, 3}, "all diffs must be 1 or 3"
    seg = re.split("3+", "".join(map(str, diff)))
    assert max(seg) == "1111", "have not hard-coded more paths"
    num_routes = {"": 1, "1": 1, "11": 2, "111": 4, "1111": 7}
    res2 = np.prod([num_routes[i] for i in seg])

    return res1, res2


def col(x, c):
    """extract column c from nested lists in x"""
    return (row[c] for row in x)


def repeat_until_stable(func, x):
    new = func(x)
    while not all(i == j for i, j in zip(new, x)):
        x = new
        new = func(x)
    return new


def day11():
    """Game of Seats."""
    x = open("11.txt").read().strip().split("\n")
    h, w = len(x), len(x[0])

    def adj(x, j, i):
        return (
            x[J][I]
            for J in range(max(j - 1, 0), min(j + 2, h))
            for I in range(max(i - 1, 0), min(i + 2, w))
            if I != i or J != j
        )

    def epoch(old):
        new = list(map(list, old))
        for j, i in product(range(h), range(w)):
            if old[j][i] == "L":
                if "#" not in adj(old, j, i):
                    new[j][i] = "#"
            elif old[j][i] == "#":
                if "".join(adj(old, j, i)).count("#") >= 4:
                    new[j][i] = "L"
        return new

    res1 = sum(i.count("#") for i in repeat_until_stable(epoch, x))

    def see(x, j, i):
        res = 0
        for line in (
            col(x[j + 1 :], i),  # S
            col(x[:j][::-1], i),  # N
            x[j][i + 1 :],  # E
            x[j][:i][::-1],  # W
            (x[J][I] for J, I in zip(range(j + 1, w), range(i + 1, h))),  # SE
            (x[J][I] for J, I in zip(range(j + 1, w), range(i - 1, -1, -1))),  # SW
            (x[J][I] for J, I in zip(range(j - 1, -1, -1), range(i + 1, h))),  # NE
            (x[J][I] for J, I in zip(range(j - 1, -1, -1), range(i - 1, -1, -1))),  # NW
        ):
            for i in line:
                if i == "#":
                    res += 1
                    break
                elif i == "L":
                    break
        return res

    def epoch2(old):
        new = list(map(list, old))
        for j, i in product(range(h), range(w)):
            if old[j][i] == "L":
                if not see(old, j, i):
                    new[j][i] = "#"
            elif old[j][i] == "#":
                if see(old, j, i) >= 5:
                    new[j][i] = "L"
        return new

    res2 = sum(i.count("#") for i in repeat_until_stable(epoch2, x))

    return res1, res2


def day12():
    """Waypoint ship navigation."""
    d = open("12.txt").read().strip().split("\n")
    d = [(i[0], int(i[1:])) for i in d]

    xy = 0  # real/imag points E/N
    o = 0  # orientation clockwise from x-axis (E)

    for k, v in d:
        if k == "F":
            k = {0: "E", 90: "N", 180: "W", 270: "S"}[o]

        if k in "NS":
            xy += v * (1j if k == "N" else -1j)
        elif k in "EW":
            xy += v * (1 if k == "E" else -1)
        elif k in "LR":
            o += v * (1 if k == "L" else -1)
            o %= 360
        else:
            raise KeyError(k)

    res1 = int(sum(map(abs, (xy.real, xy.imag))))

    wxy = 10 + 1j  # waypoint
    xy = 0  # ship
    for k, v in d:
        if k == "F":
            xy += v * wxy
        elif k in "NSEW":
            wxy += v * {"N": 1j, "S": -1j, "E": 1, "W": -1}[k]
        elif k in "LR":
            v = v * (1 if k == "L" else -1)
            wxy *= 1j ** (v / 90)
        else:
            raise KeyError(k)

    res2 = int(sum(map(abs, (xy.real, xy.imag))))

    return res1, res2


def euclid_ext(a, b):
    """
    linear combination such that `gcd(a, b) == a * x + b * y`
    Returns: dict{gcd, x, y}
    """
    aO, bO = a, b

    x = lasty = 0
    y = lastx = 1
    while b != 0:
        q = a // b
        a, b = b, a % b
        x, lastx = lastx - q * x, x
        y, lasty = lasty - q * y, y

    return {"x": lastx, "y": lasty, "gcd": aO * lastx + bO * lasty}


def linear_congruence(rem_mods):
    """
    >>> linear_congruence([(4, 19), (12, 37), (14, 43)])
    (22804, 30229)
    """
    M = reduce(lambda x, y: x * y, (i for _, i in rem_mods))
    x = 0
    for remi, modi in rem_mods:
        Mi = M // modi
        x += remi * euclid_ext(Mi, modi)["x"] * Mi
    return ((x % M) + M) % M, M


def day13():
    """Bus timetables."""
    x = open("13.txt").read().strip().split("\n")
    start = int(x[0])
    nums = {int(n) for n in x[1].split(",") if n != "x"}

    res1 = next(
        deps[0] * (i - start)
        for i in range(start, start + min(nums))
        for deps in [{i % n: n for n in nums}]
        if 0 in deps
    )

    res2 = linear_congruence(
        [(int(n) - t, int(n)) for t, n in enumerate(x[1].split(",")) if n != "x"]
    )[0]

    return res1, res2


def day14():
    """Bit masks."""
    x = open("14.txt").read().strip().split("\n")

    mem = defaultdict(int)
    msk = [(1 << 36) - 1, 0]  # zeros, ones
    for i in x:
        if i.startswith("mask"):
            msk[0] = int(i[7:].replace("X", "1"), 2)
            msk[1] = int(i[7:].replace("X", "0"), 2)
        else:
            addr, val = map(int, re.match(r"^mem\[(\d+)\] = (\d+)$", i).groups())
            mem[addr] = (val & msk[0]) | msk[1]
    res1 = sum(mem.values())

    mem = defaultdict(int)
    msk = [(1 << 36) - 1, 0, []]  # zeros, ones, floats
    for i in x:
        if i.startswith("mask"):
            msk[0] = int(i[7:].replace("0", "1").replace("X", "0"), 2)
            msk[1] = int(i[7:].replace("X", "0"), 2)
            msk[2] = [bit for bit, v in enumerate(i[7:][::-1]) if v == "X"]
        else:
            addr, val = map(int, re.match(r"^mem\[(\d+)\] = (\d+)$", i).groups())
            addr = (addr & msk[0]) | msk[1]
            for nbits in range(len(msk[2]) + 1):
                for bits in combinations(msk[2], nbits):
                    a = addr
                    for i in bits:
                        a |= 1 << i
                    mem[a] = val
    res2 = sum(mem.values())

    return res1, res2


def day15():
    """Memory Sequences."""
    x = list(map(int, open("15.txt").read().strip().split(",")))

    # defaultdict(list) takes more memory but is slightly quicker
    d = defaultdict(partial(deque, maxlen=2))
    for i, v in enumerate(x):
        d[v].append(i)

    last = x[-1]
    for i in trange(len(x), 30000000, unit_scale=True):
        last = d[last][-1] - d[last][-2] if len(d[last]) > 1 else 0
        d[last].append(i)
        if i == 2019:
            res1 = last
    res2 = last

    return res1, res2


def day16():
    """Inferring keys from valid value rules."""
    re_sub = partial(re.compile(r"^(\d)", flags=re.M).sub, r"  \1")
    d = yaml.safe_load(
        re_sub(
            open("16.txt").read().strip().replace("\n\n", "\n").replace(":\n", ": |\n")
        )
    )
    your = list(map(int, d.pop("your ticket").strip().split(",")))
    near = [
        list(map(int, i.split(",")))
        for i in d.pop("nearby tickets").strip().split("\n")
    ]

    valid = {}
    for k, i in d.items():
        valid[k] = set()
        for rule in i.split(" or "):
            a, b = map(int, rule.split("-"))
            valid[k] |= set(range(a, b + 1))

    all_valid = set()
    for i in valid.values():
        all_valid |= i
    res1 = sum(i for t in near for i in t if i not in all_valid)

    near = np.asanyarray([t for t in near if len(set(t) - all_valid) == 0])

    idxs_possible = defaultdict(list)
    for k, v in valid.items():
        for i in range(near.shape[1]):
            if len(set(near[:, i]) - v) == 0:
                idxs_possible[i].append(k)
    idxs_possible = sorted(idxs_possible.items(), key=lambda x: len(x[1]))

    idxs = {}
    seen = set()
    for i, vs in idxs_possible:
        v = (set(vs) - seen).pop()
        seen.add(v)
        idxs[v] = i

    res2 = np.prod([your[v] for k, v in idxs.items() if k.startswith("departure")])
    return res1, res2


conv = partial(convolve, mode="constant")


def day17():
    """Game of Life 4D."""
    x = np.asanyarray(
        [
            [
                [1 if i == "#" else 0 for i in j]
                for j in open("17.txt").read().strip().split("\n")
            ]
        ],
        dtype=np.int8,
    )
    x = np.pad(x, ((6, 6), (6, 6), (6, 6)))
    knl = np.ones((3, 3, 3), dtype=x.dtype)
    knl[1, 1, 1] = 0

    def epoch(old):
        adj = conv(old, knl)
        new = (((adj == 2) | (adj == 3)) & (old == 1)) | ((old == 0) & (adj == 3))
        return new.astype(old.dtype)

    res1 = x
    for _ in range(6):
        res1 = epoch(res1)
    res1 = res1.sum()

    res2 = np.pad([x], ((6, 6),) + ((0, 0),) * 3)
    knl = np.pad([knl], ((1, 1),) + ((0, 0),) * 3)
    knl[::2] = 1
    for _ in range(6):
        res2 = epoch(res2)
    res2 = res2.sum()

    return res1, res2


class IntDay18(int):
    def __add__(self, i):
        return IntDay18(int.__add__(self, i))

    def __sub__(self, i):
        """`int.__mul__` with same precendence as `__add__`"""
        return IntDay18(int.__mul__(self, i))

    def __mul__(self, i):
        """`__add__` with same precendence as `__mul__`"""
        return self + i


def day18():
    """Changing mathematical operator precedence."""
    x = open("18.txt").read().strip()
    x = re.sub("([0-9]+)", r"IntDay18(\1)", x, flags=re.M).split("\n")
    res1 = sum(eval(i.replace("*", "-")) for i in x)
    res2 = sum(eval(i.replace("*", "-").replace("+", "*")) for i in x)
    return res1, res2


def day19():
    """Matching rules."""
    rules, d = open("19.txt").read().strip().split("\n\n")
    invalid_char = "x"
    assert invalid_char not in d
    rules = yaml.safe_load(rules)
    d = d.split("\n")

    def str2int(i):
        return int(i) if re.match("^[0-9]+$", i) else i

    # tuple -> sequence, list -> OR
    rules = {
        k: (
            [tuple(map(str2int, i.split(" "))) for i in v.split(" | ")]
            if hasattr(v, "split")
            else ((v,),)
        )
        for k, v in rules.items()
    }

    maxdepth = max(map(len, d))

    def rec(node, depth=0):
        if depth > maxdepth:
            return invalid_char
        if isinstance(node, str):
            return node
        if isinstance(node, int):
            return rules[node]
        res = type(node)(map(partial(rec, depth=depth + 1), node))
        return res[0] if len(res) == 1 else res

    def flatten(node):
        if isinstance(node, str):
            return node
        if all(isinstance(i, str) for i in node):
            # tuple -> sequence, list -> OR
            return "(" + ("|" if isinstance(node, list) else "").join(node) + ")"
        return type(node)(map(flatten, node))

    def filter_rules(d):
        reg = re.compile(
            "^" + repeat_until_stable(flatten, repeat_until_stable(rec, rules[0])) + "$"
        )
        return filter(reg.match, d)

    res1 = len(list(filter_rules(d)))

    rules[8] = [(42,), (42, 8)]
    rules[11] = [(42, 31), (42, 11, 31)]
    res2 = len(list(filter_rules(d)))

    return res1, res2


def day20():
    """Reconstructing image from pieces & object detection."""
    x = open("20.txt").read().strip().split("\n\n")
    obj, x = x[0].split(":\n")[1], x[1:]
    obj = np.array(
        [[1 if i == "#" else 0 for i in row] for row in obj.split("\n")], dtype=np.uint8
    )
    x = {
        int(num[5:]): np.array(
            [[1 if i == "#" else 0 for i in row] for row in dat.split()], dtype=np.uint8
        )
        for tile in x
        for num, dat in [tile.split(":\n")]
    }

    def orientations(tile):
        return (v for tT in [tile, tile.T] for v in [np.rot90(tT, i) for i in range(4)])

    def edge_orientations(tile):
        """representations of [left, right, top, bottom] edges"""
        return [
            int("".join(map(str, i)), 2)
            for v in orientations(tile)
            for i in [v[:, 0], v[:, -1], v[0], v[-1]]  # LRTB
        ]

    edges = {k: edge_orientations(tile) for k, tile in x.items()}

    edge_links = defaultdict(set)
    for k, es in edges.items():
        for e in es:
            edge_links[e].add(k)
    g = nx.Graph()
    for v in edge_links.values():
        assert len(v) in [1, 2]
        if len(v) == 2:
            g.add_edge(*v)
    g = g.subgraph(g.nodes)

    W = int(len(x) ** 0.5)
    grid = np.zeros((W, W), dtype=int)
    coord = nx.kamada_kawai_layout(
        g, center=((W - 1) / 2, (W - 1) / 2), scale=(W - 1) / 2
    )
    for k, ij in coord.items():
        i, j = np.round(ij, 0).astype(int)
        grid[j, i] = k
    res1 = np.prod([grid[j, i] for j, i in product([0, -1], [0, -1])])

    img = np.empty((W * 8, W * 8), dtype=np.uint8)
    for j, i in product(range(W), range(W)):
        es = edges[grid[j, i]]
        valid = np.ones(len(es) // 4, dtype=bool)
        if 0 < i:  # left
            assert {grid[j, i], grid[j, i - 1]} in edge_links.values()
            valid &= [e in edges[grid[j, i - 1]] for e in es[::4]]
        if i < W - 1:  # right
            assert {grid[j, i], grid[j, i + 1]} in edge_links.values()
            valid &= [e in edges[grid[j, i + 1]] for e in es[1::4]]
        if 0 < j:  # top
            assert {grid[j, i], grid[j - 1, i]} in edge_links.values()
            valid &= [e in edges[grid[j - 1, i]] for e in es[2::4]]
        if j < W - 1:  # bottom
            assert {grid[j, i], grid[j + 1, i]} in edge_links.values()
            valid &= [e in edges[grid[j + 1, i]] for e in es[3::4]]
        assert sum(valid) == 1
        valid = np.where(valid)[0][0]
        tile = next(islice(orientations(x[grid[j, i]]), valid, valid + 1))
        img[j * 8 : (j + 1) * 8, i * 8 : (i + 1) * 8] = tile[1:-1, 1:-1]
    for i in orientations(img):
        detected = conv(i, obj) == obj.sum()
        if any(detected.flat):
            res2 = int(i.sum() - detected.sum() * obj.sum())
            break

    return res1, res2


def is_iter(x):
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes))


def flat(arr, dtype=lambda x: x):
    """
    dtype (callable): e.g. list, tuple, (default: generator)
    >>> flat([1, [2, 3, [4, 5], [6], 7], ["8.00"]], tuple)
    (1, 2, 3, 4, 5, 6, 7, '8.00')
    """
    if is_iter(arr):
        return dtype(chain(*(i if is_iter(i) else [i] for i in map(flat, arr))))
    return arr


def day21():
    """Ingredient allergens."""
    d = open("21.txt").read().strip().split("\n")
    allergens = {}
    ingredients = Counter()
    for i in d:
        ings, alls = i[:-1].split(" (contains ")
        ings = ings.split()
        ingredients.update(ings)
        ings = set(ings)
        for a in alls.split(", "):
            i = allergens.setdefault(a, ings.copy())
            i &= ings
    safe = set(ingredients) - flat(allergens.values(), set)
    res1 = sum(v for k, v in ingredients.items() if k in safe)

    bad = [v for k, v in sorted(allergens.items())]
    while any(len(i) > 1 for i in bad):
        singles = flat(filter(lambda i: len(i) == 1, bad), set)
        for d in filter(lambda i: len(i) > 1, bad):
            d -= singles
    res2 = ",".join(flat(bad))

    return res1, res2


def day22():
    """Recursive card games."""
    players = [
        deque(map(int, i.split("\n")[1:]))
        for i in open("22.txt").read().strip().split("\n\n")
    ]

    def subgame(p1, p2, rec):
        seen = set()
        while bool(p1) and bool(p2):
            state = tuple(p1), tuple(p2)
            if state in seen:
                return True
            seen.add(state)
            c1, c2 = p1.popleft(), p2.popleft()
            if rec >= 0 and len(p1) >= c1 and len(p2) >= c2:
                win1 = subgame(
                    deque(islice(p1, 0, c1)), deque(islice(p2, 0, c2)), rec + 1
                )
            else:
                win1 = c1 > c2
            if win1:
                p1.extend([c1, c2])
            else:
                p2.extend([c2, c1])
        # p1 result (if recursing) or winning score (if not recursing)
        return (
            bool(p1)
            if rec > 0
            else sum(np.prod(list(enumerate(reversed(p1 or p2), 1)), axis=1))
        )

    res1 = subgame(players[0].copy(), players[1].copy(), -1)
    res2 = subgame(players[0], players[1], 0)

    return res1, res2


def day23():
    """Circular linked list reordering."""
    d = [int(i) - 1 for i in open("23.txt").read().strip()]
    assert set(d) == set(range(len(d)))

    x = [None] * len(d)  # linked list
    for cur, nxt in zip(d, d[1:] + d[:1]):
        x[cur] = nxt

    def gen(x, start=0):
        i = x[start]
        for _ in range(len(x) - 1):
            yield i
            i = x[i]

    def run(epochs, start, progress=False):
        lab = start
        N = len(x)
        for _ in trange(epochs, disable=not progress, unit_scale=True):
            nxt1 = x[lab]
            nxt2 = x[nxt1]
            nxt3 = x[nxt2]
            dst = (lab - 1) % N
            while dst in (nxt1, nxt2, nxt3):
                dst = (dst - 1) % N
            x[lab], x[dst], x[nxt3] = x[nxt3], nxt1, x[dst]
            lab = x[lab]
        return gen(x)

    res1 = int("".join(str(i + 1) for i in run(100, d[0])))

    N = int(1e6)
    x = list(range(1, N + 1))
    for cur, nxt in zip(d, d[1:]):
        x[cur] = nxt
    x[d[-1]], x[-1] = len(d), d[0]
    res2 = run(int(1e7), d[0], True)
    res2 = np.prod([next(res2) + 1 for _ in range(2)])

    return res1, res2


def day24():
    """Game of Hexagons."""
    d = [
        Counter(re.findall(r"([ns]?[ew])", i))
        for i in open("24.txt").read().strip().split()
    ]
    yx_counts = Counter(
        (
            i["ne"] + i["nw"] - i["sw"] - i["se"],
            (i["e"] - i["w"]) * 2 + i["ne"] + i["se"] - i["nw"] - i["sw"],
        )
        for i in d
    )
    yx = np.array([k for k, v in yx_counts.items() if v % 2], dtype=int)
    res1 = len(yx)

    knlAdj = np.zeros((3, 5), dtype=np.int8)
    knlAdj[1, [0, 4]] = 1  # w e
    knlAdj[2, [1, 3]] = 1  # nw ne
    knlAdj[0, [1, 3]] = 1  # sw se

    im = np.zeros(yx.ptp(axis=0) + 1, dtype=np.int8)
    yx -= yx.min(axis=0)
    im[yx[:, 0], yx[:, 1]] = 1
    epochs = 100
    im = np.pad(im, ((epochs, epochs), (2 * epochs, 2 * epochs)))

    for _ in range(epochs):
        adj = conv(im, knlAdj)
        im[(im == 1) & ((adj == 0) | (adj > 2))] = 0
        im[(im == 0) & (adj == 2)] = 1
    res2 = im.sum()

    return res1, res2


def day25():
    """Decrypting public keys."""
    cpub, dpub = map(int, open("25.txt").read().strip().split())

    def transform(subj, loop_size):
        val = 1
        for _ in range(loop_size):
            val = (val * subj) % 20201227
        return val

    def itransform(subj, target):
        val = 1
        for i in count(1):
            val = (val * subj) % 20201227
            if val == target:
                return i

    cloop = itransform(7, cpub)
    dloop = itransform(7, dpub)
    res1 = transform(dpub, cloop)
    res2 = transform(cpub, dloop)
    assert res1 == res2

    return res1, res2

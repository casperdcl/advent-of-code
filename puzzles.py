#!/usr/bin/env python3
"""Usage:
  puzzle [<day>...]

Arguments:
  <day>  : [default: 1:int]
"""
import re
from textwrap import dedent

import numpy as np
from argopt import argopt

parser = argopt(__doc__)


def day1():
    """
    Pairs/triplets of numbers which sum to 2020.
    Returns product of answers.
    """
    x = np.loadtxt("1.txt", dtype=np.int16)
    set_x = set(x)

    y = set(2020 - x)
    res1 = np.product(list(set_x & y))

    for target in y:
        z = set(target - x)
        res2 = set_x & z
        if res2:
            res2 = list(res2)
            res2.append(2020 - sum(res2))
            assert res2[-1] in x
            res2 = np.product(res2)
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


if __name__ == "__main__":
    args = parser.parse_args()
    for day in [args.day] if isinstance(args.day, int) else args.day:
        func = globals()[f"day{day:d}"]
        print(day, dedent(func.__doc__).strip(), func())

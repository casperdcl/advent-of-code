#!/usr/bin/env python3
"""Usage:
  puzzle [<day>...]

Arguments:
  <day>  : [default: 1:int]
"""
from argopt import argopt
import numpy as np

parser = argopt(__doc__)


def day1():
    """Pairs and triplets of numbers which sum to 2020.
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


if __name__ == "__main__":
    args = parser.parse_args()
    for day in [args.day] if isinstance(args.day, int) else args.day:
        print(globals()[f"day{day:d}"]())

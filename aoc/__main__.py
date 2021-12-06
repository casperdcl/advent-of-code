#!/usr/bin/env python3
"""Usage:
  aoc [options] [<day>...]

Options:
  -y YEAR, --year YEAR  : [default: 2021:int]

Arguments:
  <day>  : [default: 0:int], use <1 for all
"""
from importlib import import_module
from os import chdir
from pathlib import Path
from textwrap import dedent
from time import time

from argopt import argopt

parser = argopt(__doc__)


def main(argv=None):
    args = parser.parse_args(argv)
    days = [args.day] if isinstance(args.day, int) else args.day
    if any(i < 1 for i in days):
        days = range(1, 26)
    sol = f"sol{args.year:d}"
    mod = import_module(f"aoc.{sol}")
    chdir(Path(__file__).parent / sol)
    for day in days:
        try:
            func = getattr(mod, f"day{day:d}")
        except AttributeError:
            break
        doc = dedent(func.__doc__).replace("\n", "\n  ").strip()
        t = time()
        print(f"{day} {doc} {func()} {time() - t:.2f}s")


if __name__ == "__main__":
    main()

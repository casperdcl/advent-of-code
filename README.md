# Advent of Code solutions

[![Tests](https://github.com/casperdcl/advent-of-code/workflows/Test/badge.svg)](https://github.com/casperdcl/advent-of-code/actions)

Solutions to problems from <https://adventofcode.com>
([2020](https://github.com/casperdcl/advent-of-code/blob/main/aoc/sol2020/__init__.py),
[2021](https://github.com/casperdcl/advent-of-code/blob/main/aoc/sol2021/__init__.py),
[2022](https://github.com/casperdcl/advent-of-code/blob/main/aoc/sol2022/__init__.py))

```sh
conda env create
conda activate aoc
python3 -m aoc --help
```

- **maintainability**
  + **readability**: style nearly production-worthy (others can understand easily; avoid amusing code golf)
  + **extensibility**: not logically-over-optimised; if requirements changed (there was a "part 3") then the code doesn't need to be rewritten
- **scalability**: time-complexity-optimised (sane run times even if the input was significantly larger; avoid brute-force just because the input is currently tiny)

## Why

Most code I've seen falls into one of these categories:

1. **novice**: e.g. logical inefficiencies, syntactical inefficiencies, unhelpful verbosity
2. **expert**: e.g. code golf, sacrificing extensibility for runtime performance, lacking code comments, unnecessary complexity, overuse of either OOP or functional programming

I strongly believe "[good code](https://xkcd.com/844)" must make none of the above sacrifices, and this repository aims to demonstrate this idea.

Suggestions and pull requests are welcome.

import sys
from os import chdir
from pathlib import Path

import pytest

sys.path.insert(1, str(Path(__file__).parent.parent))


@pytest.mark.parametrize("day", range(1, 26))
@pytest.mark.parametrize("year", range(2020, 2021 + 1))
def test_year(year, day):
    sol = f"sol{year:d}"
    mod = pytest.importorskip(f"aoc.{sol}")
    chdir(Path(__file__).parent.parent / "aoc" / sol)
    try:
        func = getattr(mod, f"day{day:d}")
    except AttributeError:
        raise pytest.skip("NotImplemented")
    print(f"{day} {func.__doc__} {func()} [.\\d+]s")

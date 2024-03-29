import re
import sys
from os import chdir
from pathlib import Path
from textwrap import dedent

import pytest

sys.path.insert(1, str(Path(__file__).parent.parent))


output = {
    2020: r"""1 Pairs/triplets of numbers which sum to 2020.
  Returns product of answers. (32064, 193598720) 0.00s
2 Number of valid passwords. (378, 280) 0.00s
3 Number of #s along a diagonal path. (171, 1206576000) 0.00s
4 Number of valid passports. (204, 179) 0.00s
5 2D binary space partitioning plane seat IDs. (913, 717) 0.00s
6 Common choices. (6662, 3382) 0.00s
7 Counting matryoshka bags. (257, 1038) 0.01s
8 Breaking infinite loops in Assembly. (2003, 1984) 0.14s
9 eXchange-Masking Addition System (XMAS). (15690279, 2174232) 0.02s
10 Chain of adapters. (1656, 56693912375296) 0.00s
11 Game of Seats. (2476, 2257) 5.02s
12 Waypoint ship navigation. (364, 39518) 0.00s
13 Bus timetables. (2845, 487905974205117) 0.00s
14 Bit masks. (17481577045893, 4160009892257) 0.04s
15 Memory Sequences. (273, 47205) 27.00s
16 Inferring keys from valid value rules. (19087, 1382443095281) 0.03s
17 Game of Life 4D. (336, 2620) 0.04s
18 Changing mathematical operator precedence. (4491283311856, 68852578641904) 0.05s
19 Matching rules. (139, 289) 4.68s
20 Reconstructing image from pieces & object detection. (2699020245973, 2012) 0.27s
21 Ingredient allergens. (2493, 'kqv,jxx,zzt,dklgl,pmvfzk,tsnkknk,qdlpbt,tlgrhdh') 0.00s
22 Recursive card games. (33473, 31793) 2.75s
23 Circular linked list reordering. (47382659, 42271866720) 6.24s
24 Game of Hexagons. (275, 3537) 0.12s
25 Decrypting public keys. (17032383, 17032383) 4.96s""",
    2021: r"""1 Number of increments. (1564, 1611) 0.01s
2 2D navigation. (1451208, 1620141160) 0.00s
3 Most common bits. (4147524, 3570354) 0.00s
4 Winning & losing Bingo. (89001, 7296) 0.00s
5 Counting line intersections. (6461, 18065) 0.15s
6 Exponential population growth. (366057, 1653559299811) 0.00s
7 Minimum total cost. (354129, 98905973) 0.00s
8 Deducing 7-segment displays. (512, 1091165) 0.01s
9 2D segmentation. (560, 959136) 0.03s
10 Brace matching errors. (311895, 2904180541) 0.00s
11 Game of flash. (1700, 273) 0.08s
12 Graph 2nd order paths. (4186, 92111) 0.50s
█    █  █ ███  ████ ███  ███  ███  █  █
█    █ █  █  █ █    █  █ █  █ █  █ █ █
█    ██   █  █ ███  ███  █  █ █  █ ██
█    █ █  ███  █    █  █ ███  ███  █ █
█    █ █  █ █  █    █  █ █    █ █  █ █
████ █  █ █  █ ████ ███  █    █  █ █  █
13 Folding paper. (653, '^plot_binary:fafa42b^') 0.01s
14 Depth First Counter. (3143, 4110215602456) 0.03s
15 Shortest path. (487, 2821) 5.01s
16 Parsing machine code. (991, 1264485568252) 0.00s
17 Discrete projectile targets. (19503, 5200) 4.02s
18 Binary Tree custom addition. (4323, 4749) 2.09s
19 3D transformed volume overlaps. (390, 13327) 135.19s
20 Game of Life infinite spawn. (5339, 18395) 0.17s
21 Counting possible outcomes. (752247, 221109915584112) 0.21s
""",
    2022: r"""1 Reducing & sorting lists. (69626, 206780) 0.00s
2 Rock, paper, scissors. (11449, 13187) 0.00s
3 Set intersections. (8515, 2434) 0.00s
4 Overlapping ranges. (433, 852) 0.00s
5 Stack manipulation. ('GFTNRBZPF', 'VRQWPDSGP') 0.00s
6 Unique sequences. (1804, 2508) 0.00s
7 Directory sizes. (1648397, 1815525) 0.00s
8 Max block reduce. (1814, 330786) 0.04s
9 Chasing 2D points. (6332, 2511) 0.48s
███   ██  █  █ ████  ██  █    █  █  ██
█  █ █  █ █  █ █    █  █ █    █  █ █  █
█  █ █    ████ ███  █    █    █  █ █
███  █ ██ █  █ █    █ ██ █    █  █ █ ██
█    █  █ █  █ █    █  █ █    █  █ █  █
█     ███ █  █ █     ███ ████  ██   ███
10 CRT drawing in assembly. (15260, '^plot_binary:90c8028^') 0.00s
11 Passing parcels. (55216, 12848882750) 0.08s
12 Shortest path. (383, 377) 0.28s
""",
}


@pytest.mark.parametrize("day", range(1, 26))
@pytest.mark.parametrize("year", output)
def test_sol(year, day, request):
    mod = pytest.importorskip(f"aoc.sol{year}")
    chdir(Path(mod.__file__).parent.resolve())
    try:
        func = getattr(mod, f"day{day:d}")
    except AttributeError:
        raise pytest.skip("NotImplemented")
    doc = re.sub(r"([^\w\s])", r"\\\1", dedent(func.__doc__).strip()).replace(
        "\n", ".*?"
    )
    expected, t = re.search(
        r"^\s*" f"{day} {doc}" r" (.*?) ([.\d]+)s$", output[year], flags=re.M | re.S
    ).groups()
    if float(t) > request.config.getoption("timeout"):
        pytest.skip("timeout")
    assert f"{func()}" == expected

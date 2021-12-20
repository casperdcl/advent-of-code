def pytest_addoption(parser):
    parser.addoption(
        "--timeout", type=int, default=60 * 60, help="Maximum test duration"
    )

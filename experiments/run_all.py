"""Canonical accelerated entry point; one shared driver for both backends."""
from run_parallel import build_parser, run


if __name__ == "__main__":
    run(build_parser().parse_args())

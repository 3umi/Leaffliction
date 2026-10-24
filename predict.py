#!/usr/bin/env python3
"""Part 4 — Predict disease for a single leaf image."""
import sys


def main(argv: list) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <image>")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

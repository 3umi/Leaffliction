"""Part 3 — Image transformations (PlantCV pipeline)."""
import sys


def main(argv: list) -> int:
    if len(argv) < 2:
        print(f"Usage: {argv[0]} <image | -src DIR -dst DIR [options]>")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

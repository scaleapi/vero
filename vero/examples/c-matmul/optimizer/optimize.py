"""Deterministic candidate producer used by CI."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    shutil.copy2(Path(__file__).with_name("optimized.c"), args.workspace / "matmul.c")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print a sample of rows from the 5m Parquet partitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Base directory containing pair=<pair>/year=<YYYY>/month=<MM>",
    )
    parser.add_argument("pair", help="Pair symbol, e.g. EURUSD")
    parser.add_argument("year", type=int, help="Year partition to inspect")
    parser.add_argument("month", type=int, help="Month partition (1-12)")
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of rows to display from the head (default: 5)",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=0,
        help="Optional number of random rows to display in addition to the head",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = (
        args.root
        / f"pair={args.pair}"
        / f"year={args.year}"
        / f"month={args.month:02d}"
        / "data.parquet"
    )
    df = pd.read_parquet(target)
    print(f"Loaded {len(df)} rows from {target}")
    print("\nHead:")
    print(df.head(args.rows).to_markdown(index=False))
    if args.random > 0:
        print("\nRandom sample:")
        print(df.sample(args.random).to_markdown(index=False))


if __name__ == "__main__":
    main()

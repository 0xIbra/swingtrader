#!/usr/bin/env python3
"""Utility to convert EURUSD 1m CSV data into 5m bars with extras."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


COLUMNS = ["timestamp", "open", "high", "low", "close", "tick_volume"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="Path to the 1m CSV file")
    parser.add_argument(
        "--pair",
        default="EURUSD",
        help="Symbol/pair name to embed in partition paths (default: EURUSD)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_5m"),
        help="Directory for Parquet partitions (default: ./data_5m)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional inclusive start timestamp filter (YYYY-MM-DD HH:MM)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional exclusive end timestamp filter (YYYY-MM-DD HH:MM)",
    )
    return parser.parse_args()


def load_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        names=COLUMNS,
        header=None,
        parse_dates=["timestamp"],
        date_parser=lambda x: pd.to_datetime(x, format="%Y%m%d %H%M%S", utc=True),
    )
    df = df.sort_values("timestamp").set_index("timestamp")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df["tick_volume"] = df["tick_volume"].astype(float)
    return df


def apply_time_filter(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index < pd.Timestamp(end, tz="UTC")]
    return df


def resample_to_five_minutes(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.resample("5T").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "tick_volume": "sum",
        }
    )
    missing_mask = agg["open"].isna()
    price_cols = ["open", "high", "low", "close"]
    agg[price_cols] = agg[price_cols].ffill()
    agg["tick_volume"] = agg["tick_volume"].fillna(0)
    agg["missing_flag"] = missing_mask.astype("int8")
    return agg


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index.tz_convert("UTC")
    hours = idx.hour
    minutes = idx.minute

    df["session_asia"] = (((hours >= 0) & (hours < 6))).astype("int8")
    df["session_london"] = (((hours >= 7) & (hours < 15))).astype("int8")
    df["session_ny"] = (((hours >= 13) & (hours < 21))).astype("int8")
    df["session_overlap"] = (
        ((hours >= 13) & (hours < 16))
    ).astype("int8")

    hour_decimal = hours + minutes / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour_decimal / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour_decimal / 24.0)

    dow = idx.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return df


def add_placeholder_spread(df: pd.DataFrame) -> pd.DataFrame:
    df["bid"] = np.nan
    df["ask"] = np.nan
    df["spread"] = np.nan
    return df


def write_partitions(df: pd.DataFrame, output_dir: Path, pair: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["timestamp"] = df.index
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["pair"] = pair

    partition_cols = [
        "timestamp",
        "pair",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "bid",
        "ask",
        "spread",
        "missing_flag",
        "session_asia",
        "session_london",
        "session_ny",
        "session_overlap",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
    df = df[partition_cols + ["year", "month"]]

    for (year, month), part in df.groupby(["year", "month"], sort=True):
        part_dir = output_dir / f"pair={pair}" / f"year={year}" / f"month={month:02d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        part_to_save = part.drop(columns=["year", "month"])
        part_to_save.to_parquet(part_dir / "data.parquet", index=False)


def main() -> None:
    args = parse_args()
    df = load_raw_csv(args.input_csv)
    df = apply_time_filter(df, args.start, args.end)
    df = resample_to_five_minutes(df)
    df = add_session_features(df)
    df = add_placeholder_spread(df)
    write_partitions(df, args.output_dir, args.pair)


if __name__ == "__main__":
    main()

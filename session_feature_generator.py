"""
Session feature generator for EURUSD hourly candles.

Loads a CSV file with at least a ``timestamp`` column, creates trading session
flags and cyclical encodings, and writes the augmented dataset to disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

PI2: Final[float] = 2.0 * np.pi


def _ensure_timestamp(series: pd.Series) -> pd.Series:
    """Return the timestamp column as timezone-aware UTC datetimes."""
    ts = pd.to_datetime(series, utc=True)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    return ts


def _session_flags(hour: pd.Series) -> pd.DataFrame:
    """Compute intraday session flags from an hour-of-day series."""
    asia = hour.between(0, 6, inclusive="both")
    london = hour.between(7, 15, inclusive="both")
    ny = hour.between(13, 21, inclusive="both")
    overlap = hour.between(13, 16, inclusive="both")

    return pd.DataFrame(
        {
            "session_asia": asia.astype("int8"),
            "session_london": london.astype("int8"),
            "session_ny": ny.astype("int8"),
            "session_overlap": overlap.astype("int8"),
        }
    )


def _cyclical_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cyclical hour-of-day and day-of-week encodings."""
    hour = df["timestamp"].dt.hour.astype(float)
    dow = df["timestamp"].dt.dayofweek.astype(float)  # Monday=0

    hour_angle = PI2 * hour / 24.0
    dow_angle = PI2 * dow / 7.0

    return pd.DataFrame(
        {
            "hour_sin": np.sin(hour_angle),
            "hour_cos": np.cos(hour_angle),
            "dow_sin": np.sin(dow_angle),
            "dow_cos": np.cos(dow_angle),
        }
    )


def build_session_features(input_path: Path, output_path: Path) -> None:
    """Load OHLC data, append session features, and save to CSV."""
    df = pd.read_csv(input_path)
    if "timestamp" not in df.columns:
        raise ValueError("Input file must contain a 'timestamp' column")

    df["timestamp"] = _ensure_timestamp(df["timestamp"])
    hour = df["timestamp"].dt.hour

    features = pd.concat(
        [
            _session_flags(hour),
            _cyclical_encodings(df),
        ],
        axis=1,
    )

    enriched = pd.concat([df, features], axis=1)
    enriched.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate session and cyclical features for hourly FX candles."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the source CSV with timestamp + OHLC columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the enriched CSV with session features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_session_features(args.input, args.output)


if __name__ == "__main__":
    main()

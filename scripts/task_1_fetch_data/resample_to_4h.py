#!/usr/bin/env python3
"""Resample 1-minute EURUSD data to 4-hour candlesticks.

This script loads the 1m data and resamples it to 4H intervals aligned to
standard 4-hour candles: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from datetime import datetime

# Configuration
INPUT_DIR = Path('data/raw/1m')
OUTPUT_DIR = Path('data/raw/4h')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resample_to_4h(df_1m):
    """
    Resample 1-minute data to 4-hour candlesticks.

    Args:
        df_1m: DataFrame with 1m OHLCV data (timestamp as column)

    Returns:
        DataFrame with 4H OHLCV data
    """
    print("🔄 Resampling 1m → 4H...")

    # Set timestamp as index for resampling
    df_1m = df_1m.copy()
    df_1m.set_index('timestamp', inplace=True)
    df_1m.index = pd.to_datetime(df_1m.index)

    # Resample to 4H with standard aggregation
    # '4H' will align to midnight by default (00:00, 04:00, 08:00, etc.)
    df_4h = df_1m.resample('4H').agg({
        'open': 'first',    # First value in 4H window
        'high': 'max',      # Maximum value
        'low': 'min',       # Minimum value
        'close': 'last',    # Last value
        'volume': 'sum'     # Sum of volumes (may be null for forex)
    })

    # Remove rows with NaN (incomplete 4H bars at the end)
    df_4h.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

    # Reset index to make timestamp a column again
    df_4h.reset_index(inplace=True)

    print(f"✓ Resampled to {len(df_4h):,} 4H candles")
    print(f"✓ Date range: {df_4h['timestamp'].min()} to {df_4h['timestamp'].max()}")

    return df_4h


def validate_4h_candles(df_4h):
    """
    Validate that 4H candles are properly aligned and have valid OHLC.

    Args:
        df_4h: DataFrame with 4H data

    Returns:
        bool: True if validation passes
    """
    print("\n🔍 Validating 4H candles...")

    issues = []

    # Check timestamp alignment (should be 00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
    valid_hours = {0, 4, 8, 12, 16, 20}
    invalid_hours = df_4h['timestamp'].dt.hour[~df_4h['timestamp'].dt.hour.isin(valid_hours)]
    if len(invalid_hours) > 0:
        issues.append(f"Found {len(invalid_hours)} candles not aligned to 4H grid")

    # Check OHLC relationships (High >= Low, High >= Open/Close, Low <= Open/Close)
    invalid_hl = df_4h[df_4h['high'] < df_4h['low']]
    if len(invalid_hl) > 0:
        issues.append(f"Found {len(invalid_hl)} candles where High < Low")

    invalid_ho = df_4h[df_4h['high'] < df_4h['open']]
    if len(invalid_ho) > 0:
        issues.append(f"Found {len(invalid_ho)} candles where High < Open")

    invalid_hc = df_4h[df_4h['high'] < df_4h['close']]
    if len(invalid_hc) > 0:
        issues.append(f"Found {len(invalid_hc)} candles where High < Close")

    invalid_lo = df_4h[df_4h['low'] > df_4h['open']]
    if len(invalid_lo) > 0:
        issues.append(f"Found {len(invalid_lo)} candles where Low > Open")

    invalid_lc = df_4h[df_4h['low'] > df_4h['close']]
    if len(invalid_lc) > 0:
        issues.append(f"Found {len(invalid_lc)} candles where Low > Close")

    # Check for gaps > 24 hours (excluding weekends)
    df_4h['time_diff'] = df_4h['timestamp'].diff()
    large_gaps = df_4h[df_4h['time_diff'] > pd.Timedelta(hours=24)]
    # Filter out weekend gaps (Friday 20:00 to Monday 00:00 is expected)
    weekday_gaps = large_gaps[
        ~((large_gaps['timestamp'].dt.dayofweek == 0) &  # Monday
          (large_gaps['timestamp'].dt.hour == 0))
    ]
    if len(weekday_gaps) > 0:
        issues.append(f"Found {len(weekday_gaps)} unexpected gaps > 24 hours")

    # Report results
    if issues:
        print("⚠️  Validation issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ All validation checks passed!")
        return True


def main():
    """
    Main function to resample 1m data to 4H.
    """
    print("=" * 80)
    print("EURUSD 1m → 4H Resampler")
    print("=" * 80)
    print()

    # Find the combined 1m file
    input_files = list(INPUT_DIR.glob('EURUSD_1m_2020_*.csv'))

    if not input_files:
        print("❌ No combined 1m file found!")
        print(f"Expected file pattern: {INPUT_DIR}/EURUSD_1m_2020_*.csv")
        print("\nPlease run fetch_eodhd_1m.py first.")
        return

    # Use the most recent file
    input_file = sorted(input_files)[-1]
    print(f"📂 Input file: {input_file.absolute()}")

    # Load 1m data
    print(f"📥 Loading 1m data...")
    df_1m = pd.read_csv(input_file)
    df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'])

    print(f"✓ Loaded {len(df_1m):,} 1m candles")
    print(f"✓ Date range: {df_1m['timestamp'].min()} to {df_1m['timestamp'].max()}")
    print()

    # Resample to 4H
    df_4h = resample_to_4h(df_1m)

    # Validate
    is_valid = validate_4h_candles(df_4h)

    # Save 4H data
    end_year = datetime.now().year
    output_file = OUTPUT_DIR / f'EURUSD_4H_2020_{end_year}.csv'

    print(f"\n💾 Saving 4H data...")
    df_4h.to_csv(output_file, index=False)

    print()
    print("=" * 80)
    print("✅ SUCCESS!")
    print(f"📊 Total 4H candles: {len(df_4h):,}")
    print(f"📅 Date range: {df_4h['timestamp'].min()} to {df_4h['timestamp'].max()}")
    print(f"💾 Output file: {output_file.absolute()}")
    print()

    # Display sample
    print("📋 First 10 rows:")
    print(df_4h.head(10))
    print()
    print("📋 Last 10 rows:")
    print(df_4h.tail(10))
    print()

    # Statistics
    print("📊 Candle Alignment Check:")
    hour_counts = df_4h['timestamp'].dt.hour.value_counts().sort_index()
    for hour, count in hour_counts.items():
        print(f"   {hour:02d}:00 UTC → {count:,} candles")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

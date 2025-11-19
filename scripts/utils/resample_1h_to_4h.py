#!/usr/bin/env python3
"""
Resample 1H EURUSD data to 4H timeframe.

4H timeframe provides:
- More stable patterns
- Longer pattern persistence
- Better signal-to-noise ratio
- Reduced overfitting risk
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def resample_ohlcv_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1H OHLCV data to 4H bars.

    Resampling rules:
    - Open: first value in 4H period
    - High: maximum value in 4H period
    - Low: minimum value in 4H period
    - Close: last value in 4H period
    - Volume: sum of values in 4H period

    Args:
        df: DataFrame with 1H OHLCV data

    Returns:
        DataFrame with 4H OHLCV data
    """
    print("\n" + "="*80)
    print("Resampling 1H → 4H")
    print("="*80)

    # Set timestamp as index for resampling
    df = df.copy()
    df.set_index('timestamp', inplace=True)

    print(f"\nOriginal 1H data:")
    print(f"  Bars: {len(df):,}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    # Define resampling rules
    resample_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    # Resample to 4H
    # Use '4H' for 4-hour intervals, label='right' to label with end of interval
    df_4h = df.resample('4h', label='right', closed='right').agg(resample_rules)

    # Remove rows where all OHLC values are NaN (no data in that 4H period)
    df_4h = df_4h.dropna(subset=['open', 'high', 'low', 'close'], how='all')

    # Reset index to make timestamp a column again
    df_4h.reset_index(inplace=True)

    print(f"\nResampled 4H data:")
    print(f"  Bars: {len(df_4h):,}")
    print(f"  Date range: {df_4h['timestamp'].min()} to {df_4h['timestamp'].max()}")
    print(f"  Reduction: {len(df) / len(df_4h):.2f}x fewer bars")

    # Validate resampling
    print(f"\n📊 Data Quality Check:")

    # Check for NaN values
    nan_counts = df_4h[['open', 'high', 'low', 'close']].isna().sum()
    if nan_counts.sum() > 0:
        print(f"  ⚠️  NaN values found:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"    - {col}: {count}")
    else:
        print(f"  ✓ No NaN values in OHLC data")

    # Check for invalid bars (high < low)
    invalid_bars = (df_4h['high'] < df_4h['low']).sum()
    if invalid_bars > 0:
        print(f"  ⚠️  Invalid bars (high < low): {invalid_bars}")
    else:
        print(f"  ✓ All bars valid (high >= low)")

    # Check for zero-range bars
    zero_range = (df_4h['high'] == df_4h['low']).sum()
    if zero_range > 0:
        print(f"  ⚠️  Zero-range bars: {zero_range} ({zero_range/len(df_4h)*100:.2f}%)")
    else:
        print(f"  ✓ No zero-range bars")

    return df_4h


def main():
    """Main execution"""

    print("="*80)
    print("EURUSD 1H → 4H Resampling")
    print("="*80)

    # Get project root directory (cross-platform)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Load 1H data
    input_file = project_root / 'data' / 'EURUSD_1H_2020_2025.csv'
    print(f"\nLoading 1H data from: {input_file}")

    df_1h = pd.read_csv(input_file)
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], utc=True)

    print(f"✓ Loaded {len(df_1h):,} 1H bars")
    print(f"✓ Columns: {list(df_1h.columns)}")

    # Resample to 4H
    df_4h = resample_ohlcv_to_4h(df_1h)

    # Save 4H data
    output_file = project_root / 'data' / 'EURUSD_4H_2020_2025.csv'
    print(f"\n📁 Saving 4H data to: {output_file}")
    df_4h.to_csv(output_file, index=False)

    file_size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"✓ Saved ({file_size_mb:.2f} MB)")

    # Display sample
    print("\n📊 Sample of 4H data (first 10 bars):")
    print(df_4h.head(10).to_string(index=False))

    print("\n📊 Sample of 4H data (last 10 bars):")
    print(df_4h.tail(10).to_string(index=False))

    # Statistics
    print("\n📊 4H Data Statistics:")
    print(df_4h[['open', 'high', 'low', 'close']].describe())

    print("\n" + "="*80)
    print("✅ RESAMPLING COMPLETE")
    print("="*80)
    print(f"✓ Output: {output_file}")
    print(f"✓ Bars: {len(df_4h):,}")
    print(f"✓ Ready for feature engineering pipeline")

    print("\n📋 Next Steps:")
    print("  1. Run session_feature_generator.py with 4H data")
    print("  2. Run economic_calendar_pipeline.py with 4H data")
    print("  3. Run news_sentiment_pipeline.py with 4H data")
    print("  4. Run macro_regime_pipeline.py with 4H data")
    print("  5. Run price_features_pipeline_4h.py (adjusted parameters)")
    print("  6. Run mfe_mae_labels_pipeline_4h.py (6-bar horizon)")
    print("  7. Run direction_labels_pipeline.py")
    print("  8. Run sequence_window_builder_4h.py (42-bar lookback)")
    print("  9. Run dataset_split_normalize.py")
    print(" 10. Train TCN model on 4H data")


if __name__ == "__main__":
    main()


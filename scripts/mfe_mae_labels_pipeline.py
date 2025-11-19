#!/usr/bin/env python3
"""
TASK 7: MFE/MAE Labels Pipeline
Compute Maximum Favorable Excursion and Maximum Adverse Excursion labels.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def compute_mfe_mae_labels(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """
    Compute MFE/MAE labels for both long and short positions.

    For each timestamp t:
    - H = future horizon (24 bars = 24 hours = 1 day)
    - entry = close[t]

    LONG:
    - MFE_long = max(high[t+1..t+H]) - entry
    - MAE_long = min(low[t+1..t+H]) - entry

    SHORT:
    - MFE_short = entry - min(low[t+1..t+H])
    - MAE_short = entry - max(high[t+1..t+H])

    Normalize by ATR(t):
    - mfe_l = MFE_long / ATR_t
    - mae_l = MAE_long / ATR_t
    - mfe_s = MFE_short / ATR_t
    - mae_s = MAE_short / ATR_t

    Args:
        df: DataFrame with OHLC and ATR columns
        horizon: Forward-looking horizon in bars (default 24 = 1 day)

    Returns:
        DataFrame with MFE/MAE labels added
    """
    print("\n" + "="*80)
    print(f"Computing MFE/MAE Labels (horizon={horizon} bars)")
    print("="*80)

    df = df.copy()

    # Initialize label columns
    df['MFE_long'] = np.nan
    df['MAE_long'] = np.nan
    df['MFE_short'] = np.nan
    df['MAE_short'] = np.nan
    df['mfe_l'] = np.nan
    df['mae_l'] = np.nan
    df['mfe_s'] = np.nan
    df['mae_s'] = np.nan

    # Compute labels for each bar (except last H bars)
    n_bars = len(df) - horizon

    print(f"\nComputing excursions for {n_bars:,} bars...")
    print(f"(Skipping last {horizon} bars without full forward horizon)")

    for i in tqdm(range(n_bars), desc="Computing MFE/MAE"):
        # Entry price
        entry = df.loc[i, 'close']
        atr = df.loc[i, 'atr_14']

        # Skip if ATR is zero or NaN
        if pd.isna(atr) or atr <= 0:
            continue

        # Future window [t+1 .. t+H]
        future_start = i + 1
        future_end = i + horizon + 1
        future_window = df.iloc[future_start:future_end]

        # Skip if future window is incomplete
        if len(future_window) < horizon:
            continue

        # Get max high and min low in future window
        max_high = future_window['high'].max()
        min_low = future_window['low'].min()

        # LONG position
        MFE_long = max_high - entry  # Best case for long
        MAE_long = min_low - entry   # Worst case for long

        # SHORT position
        MFE_short = entry - min_low  # Best case for short
        MAE_short = entry - max_high # Worst case for short

        # Store raw values
        df.loc[i, 'MFE_long'] = MFE_long
        df.loc[i, 'MAE_long'] = MAE_long
        df.loc[i, 'MFE_short'] = MFE_short
        df.loc[i, 'MAE_short'] = MAE_short

        # Normalize by ATR
        df.loc[i, 'mfe_l'] = MFE_long / atr
        df.loc[i, 'mae_l'] = MAE_long / atr
        df.loc[i, 'mfe_s'] = MFE_short / atr
        df.loc[i, 'mae_s'] = MAE_short / atr

    # Report statistics
    print("\n" + "="*80)
    print("Label Statistics")
    print("="*80)

    labels_computed = df['mfe_l'].notna().sum()
    labels_total = len(df)

    print(f"\nLabels computed: {labels_computed:,} / {labels_total:,} ({labels_computed/labels_total*100:.2f}%)")
    print(f"Labels missing (last {horizon} bars): {labels_total - labels_computed:,}")

    # Statistics for normalized labels
    print("\n📊 Normalized MFE/MAE Statistics (in ATR units):")
    print("\nLONG positions:")
    print(f"  MFE (favorable): mean={df['mfe_l'].mean():.4f}, std={df['mfe_l'].std():.4f}, "
          f"min={df['mfe_l'].min():.4f}, max={df['mfe_l'].max():.4f}")
    print(f"  MAE (adverse):   mean={df['mae_l'].mean():.4f}, std={df['mae_l'].std():.4f}, "
          f"min={df['mae_l'].min():.4f}, max={df['mae_l'].max():.4f}")

    print("\nSHORT positions:")
    print(f"  MFE (favorable): mean={df['mfe_s'].mean():.4f}, std={df['mfe_s'].std():.4f}, "
          f"min={df['mfe_s'].min():.4f}, max={df['mfe_s'].max():.4f}")
    print(f"  MAE (adverse):   mean={df['mae_s'].mean():.4f}, std={df['mae_s'].std():.4f}, "
          f"min={df['mae_s'].min():.4f}, max={df['mae_s'].max():.4f}")

    # Analysis: How often is MFE > 0?
    long_favorable_pct = (df['mfe_l'] > 0).sum() / labels_computed * 100
    short_favorable_pct = (df['mfe_s'] > 0).sum() / labels_computed * 100

    print("\n📈 Market Opportunity Analysis:")
    print(f"  - LONG trades with MFE > 0:  {long_favorable_pct:.2f}%")
    print(f"  - SHORT trades with MFE > 0: {short_favorable_pct:.2f}%")

    # Distribution of normalized excursions
    print("\n📊 MFE Distribution (ATR units):")
    mfe_bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, float('inf')]
    mfe_labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-3.0', '3.0-5.0', '5.0+']

    for label_col, position in [('mfe_l', 'LONG'), ('mfe_s', 'SHORT')]:
        print(f"\n  {position}:")
        dist = pd.cut(df[label_col].dropna(), bins=mfe_bins, labels=mfe_labels)
        counts = dist.value_counts(sort=False)
        for bin_label, count in counts.items():
            pct = count / labels_computed * 100
            print(f"    {bin_label} ATR: {count:6,} ({pct:5.2f}%)")

    return df


def main():
    """Main pipeline execution"""

    print("="*80)
    print("TASK 7: MFE/MAE Labels Pipeline")
    print("="*80)

    # Get project root directory (cross-platform)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Load data with price features
    input_file = project_root / 'data' / 'EURUSD_1H_2020_2025_with_price_features.csv'
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    print(f"✓ Loaded {len(df):,} price bars")
    print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"✓ Current feature count: {len(df.columns)}")

    # Verify required columns exist
    required_cols = ['close', 'high', 'low', 'atr_14']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"\n❌ Error: Missing required columns: {missing_cols}")
        return

    print(f"✓ All required columns present: {required_cols}")

    # Compute MFE/MAE labels with horizon = 24 bars (24 hours = 1 day)
    HORIZON = 24
    df = compute_mfe_mae_labels(df, horizon=HORIZON)

    # Save enhanced data
    output_file = project_root / 'data' / 'EURUSD_1H_2020_2025_with_labels.csv'
    df.to_csv(output_file, index=False)

    print("\n" + "="*80)
    print("✅ TASK 7 COMPLETE")
    print("="*80)
    print(f"✓ Output file: {output_file}")
    print(f"✓ Total rows: {len(df):,}")
    print(f"✓ Total features: {len(df.columns)}")

    print(f"\n✓ New label columns added (8):")
    label_cols = ['MFE_long', 'MAE_long', 'MFE_short', 'MAE_short',
                  'mfe_l', 'mae_l', 'mfe_s', 'mae_s']
    for col in label_cols:
        print(f"  - {col}")

    print("\n📊 Sample of labels (first 10 rows with labels):")
    sample_cols = ['timestamp', 'close', 'atr_14', 'mfe_l', 'mae_l', 'mfe_s', 'mae_s']
    sample_df = df[df['mfe_l'].notna()][sample_cols].head(10)
    print(sample_df.to_string(index=False))

    print("\n📊 Sample of labels (last 10 rows with labels):")
    sample_df = df[df['mfe_l'].notna()][sample_cols].tail(10)
    print(sample_df.to_string(index=False))


if __name__ == "__main__":
    main()


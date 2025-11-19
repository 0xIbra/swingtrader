#!/usr/bin/env python3
"""
TASK 7: MFE/MAE Labels Pipeline (4H Timeframe)
Future excursion label generator for 4H bars.

Adjusted parameters:
- Horizon: 6 bars (24 hours at 4H timeframe)
- ATR normalization same as before
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def compute_mfe_mae_labels(
    df: pd.DataFrame,
    horizon: int = 6  # 6 bars = 24 hours at 4H
) -> pd.DataFrame:
    """
    Compute MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion).

    For each timestamp t:
    - Look ahead H bars (horizon)
    - Compute best/worst price moves for both LONG and SHORT
    - Normalize by ATR for regime-invariance

    Args:
        df: DataFrame with OHLC and ATR columns
        horizon: Number of bars to look ahead (6 bars = 24h at 4H)

    Returns:
        DataFrame with MFE/MAE labels added
    """
    print("\n" + "="*80)
    print(f"Computing MFE/MAE Labels (4H Timeframe)")
    print(f"  Horizon: {horizon} bars ({horizon * 4} hours)")
    print("="*80)

    df = df.copy()

    # Initialize columns
    df['MFE_long'] = np.nan
    df['MAE_long'] = np.nan
    df['MFE_short'] = np.nan
    df['MAE_short'] = np.nan
    df['mfe_l'] = np.nan
    df['mae_l'] = np.nan
    df['mfe_s'] = np.nan
    df['mae_s'] = np.nan

    # Only compute for bars where we have sufficient future data
    valid_indices = range(len(df) - horizon)

    print(f"\nComputing labels for {len(valid_indices):,} bars...")
    print(f"(Last {horizon} bars skipped - insufficient future data)")

    for i in tqdm(valid_indices, desc="Processing bars"):
        entry = df.loc[i, 'close']
        atr = df.loc[i, 'atr_14']

        # Skip if ATR is invalid
        if pd.isna(atr) or atr <= 0:
            continue

        # Get future window [t+1 : t+horizon]
        future_window = df.iloc[i+1 : i+horizon+1]

        # Skip if window has missing data
        if future_window[['high', 'low']].isna().any().any():
            continue

        future_highs = future_window['high'].values
        future_lows = future_window['low'].values

        # LONG position
        # MFE = maximum profit = highest high - entry
        # MAE = maximum loss = lowest low - entry (will be negative)
        MFE_long = np.max(future_highs) - entry
        MAE_long = np.min(future_lows) - entry

        # SHORT position
        # MFE = maximum profit = entry - lowest low
        # MAE = maximum loss = entry - highest high (will be negative)
        MFE_short = entry - np.min(future_lows)
        MAE_short = entry - np.max(future_highs)

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

    # Count labeled bars
    labeled_count = df['mfe_l'].notna().sum()

    print(f"\n✓ Labels computed for {labeled_count:,} bars")
    print(f"  ({labeled_count / len(df) * 100:.2f}% of total)")

    return df


def analyze_labels(df: pd.DataFrame) -> None:
    """Analyze MFE/MAE label statistics."""
    print("\n" + "="*80)
    print("MFE/MAE Label Statistics")
    print("="*80)

    labeled_df = df[df['mfe_l'].notna()].copy()

    print(f"\n📊 Raw Values (in price units):")
    raw_cols = ['MFE_long', 'MAE_long', 'MFE_short', 'MAE_short']
    print(labeled_df[raw_cols].describe())

    print(f"\n📊 ATR-Normalized Values:")
    norm_cols = ['mfe_l', 'mae_l', 'mfe_s', 'mae_s']
    print(labeled_df[norm_cols].describe())

    # Analyze profitability
    print(f"\n📊 Profitability Analysis:")

    long_profitable = (labeled_df['mfe_l'] > abs(labeled_df['mae_l'])).sum()
    short_profitable = (labeled_df['mfe_s'] > abs(labeled_df['mae_s'])).sum()

    print(f"  LONG favorable:  {long_profitable:,} / {len(labeled_df):,} ({long_profitable/len(labeled_df)*100:.2f}%)")
    print(f"  SHORT favorable: {short_profitable:,} / {len(labeled_df):,} ({short_profitable/len(labeled_df)*100:.2f}%)")

    # Distribution by range
    print(f"\n📊 MFE Distribution (ATR-normalized):")
    mfe_ranges = [
        ('Very Low (< 0.5 ATR)', labeled_df['mfe_l'] < 0.5),
        ('Low (0.5-1.0 ATR)', labeled_df['mfe_l'].between(0.5, 1.0)),
        ('Medium (1.0-2.0 ATR)', labeled_df['mfe_l'].between(1.0, 2.0)),
        ('High (2.0-3.0 ATR)', labeled_df['mfe_l'].between(2.0, 3.0)),
        ('Very High (> 3.0 ATR)', labeled_df['mfe_l'] > 3.0)
    ]

    for label, mask in mfe_ranges:
        count = mask.sum()
        pct = count / len(labeled_df) * 100
        print(f"  {label:25s}: {count:6,} ({pct:5.2f}%)")

    # Temporal analysis
    print(f"\n📊 Temporal Analysis:")
    labeled_df['year'] = pd.to_datetime(labeled_df['timestamp']).dt.year
    yearly_stats = labeled_df.groupby('year')[['mfe_l', 'mae_l', 'mfe_s', 'mae_s']].mean()

    print("\n  Yearly averages (ATR-normalized):")
    print(yearly_stats.round(3))


def main():
    """Main pipeline execution"""

    print("="*80)
    print("TASK 7: MFE/MAE Labels Pipeline (4H)")
    print("="*80)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Load data with price features
    input_file = project_root / 'data' / 'EURUSD_4H_2020_2025_with_price_features.csv'
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    print(f"✓ Loaded {len(df):,} 4H bars")
    print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Verify ATR column exists
    if 'atr_14' not in df.columns:
        print(f"\n❌ Error: 'atr_14' column not found")
        print(f"   Run price_features_pipeline_4h.py first")
        return

    print(f"✓ ATR column present")

    # Compute MFE/MAE labels
    HORIZON = 6  # 6 bars = 24 hours at 4H
    df = compute_mfe_mae_labels(df, horizon=HORIZON)

    # Analyze labels
    analyze_labels(df)

    # Save
    output_file = project_root / 'data' / 'EURUSD_4H_2020_2025_with_labels.csv'
    df.to_csv(output_file, index=False)

    print("\n" + "="*80)
    print("✅ TASK 7 COMPLETE (4H)")
    print("="*80)
    print(f"✓ Output: {output_file}")
    print(f"✓ Total bars: {len(df):,}")
    print(f"✓ Labeled bars: {df['mfe_l'].notna().sum():,}")

    print(f"\n✓ New label columns (8):")
    label_cols = ['MFE_long', 'MAE_long', 'MFE_short', 'MAE_short',
                  'mfe_l', 'mae_l', 'mfe_s', 'mae_s']
    for col in label_cols:
        print(f"  - {col}")

    print("\n📊 Sample (first 10 labeled rows):")
    sample_df = df[df['mfe_l'].notna()][['timestamp', 'close', 'atr_14', 'mfe_l', 'mae_l', 'mfe_s', 'mae_s']].head(10)
    print(sample_df.to_string(index=False))


if __name__ == "__main__":
    main()


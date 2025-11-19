#!/usr/bin/env python3
"""
Add economic events, sentiment, and macro features to 4H data.
Resample these features from the 1H dataset.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def resample_features_to_4h(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Resample features from 1H to 4H data.

    Strategy:
    - Event features: use last value in 4H period (most recent)
    - Sentiment features: average over 4H period
    - Macro features: last value in 4H period
    - Session features: already computed based on timestamp
    """
    print("\n" + "="*80)
    print("Resampling Features from 1H to 4H")
    print("="*80)

    # Identify feature categories
    event_features = [c for c in df_1h.columns if 'event' in c or 'hours_' in c]
    sentiment_features = [c for c in df_1h.columns if 'sent_' in c]
    macro_features = [c for c in df_1h.columns if any(m in c for m in
                     ['spx_', 'vix_', 'yield10_', 'gold_', 'oil_', 'dxy_'])]

    print(f"\nFeature groups to resample:")
    print(f"  - Event features: {len(event_features)}")
    print(f"  - Sentiment features: {len(sentiment_features)}")
    print(f"  - Macro features: {len(macro_features)}")

    # Set timestamp as index for resampling
    df_1h_indexed = df_1h.set_index('timestamp')

    # Prepare result DataFrame
    df_result = df_4h.copy()

    # Resample event features (use last value)
    if event_features:
        print("\n📊 Resampling event features (last)...")
        event_resampled = df_1h_indexed[event_features].resample('4h', label='right', closed='right').last()
        event_resampled = event_resampled.reset_index()

        # Merge with 4H data
        for feat in event_features:
            df_result[feat] = df_result['timestamp'].map(
                dict(zip(event_resampled['timestamp'], event_resampled[feat]))
            )
            print(f"  ✓ {feat}")

    # Resample sentiment features (average)
    if sentiment_features:
        print("\n📊 Resampling sentiment features (mean)...")
        sentiment_resampled = df_1h_indexed[sentiment_features].resample('4h', label='right', closed='right').mean()
        sentiment_resampled = sentiment_resampled.reset_index()

        # Merge with 4H data
        for feat in sentiment_features:
            df_result[feat] = df_result['timestamp'].map(
                dict(zip(sentiment_resampled['timestamp'], sentiment_resampled[feat]))
            )
            print(f"  ✓ {feat}")

    # Resample macro features (last value)
    if macro_features:
        print("\n📊 Resampling macro features (last)...")
        macro_resampled = df_1h_indexed[macro_features].resample('4h', label='right', closed='right').last()
        macro_resampled = macro_resampled.reset_index()

        # Merge with 4H data
        for feat in macro_features:
            df_result[feat] = df_result['timestamp'].map(
                dict(zip(macro_resampled['timestamp'], macro_resampled[feat]))
            )
            print(f"  ✓ {feat}")

    # Fill NaN values
    print("\n📊 Handling NaN values...")
    all_features = event_features + sentiment_features + macro_features

    for feat in all_features:
        if feat in df_result.columns:
            # Forward fill then backward fill
            df_result[feat] = df_result[feat].ffill().bfill()

            # If still NaN, fill with 0
            if df_result[feat].isna().any():
                df_result[feat] = df_result[feat].fillna(0)

    print("  ✓ All NaN values handled")

    return df_result


def main():
    """Main execution"""

    print("="*80)
    print("Add Features to 4H Data")
    print("="*80)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Load 1H data with all features (before price features)
    print("\nLoading 1H data with events/sentiment/macro...")
    df_1h_file = project_root / 'data' / 'EURUSD_1H_2020_2025_with_macro.csv'
    df_1h = pd.read_csv(df_1h_file)
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], utc=True)

    print(f"✓ Loaded {len(df_1h):,} 1H bars")
    print(f"✓ Columns: {len(df_1h.columns)}")

    # Load 4H data with sessions
    print("\nLoading 4H data with sessions...")
    df_4h_file = project_root / 'data' / 'EURUSD_4H_2020_2025_with_sessions.csv'
    df_4h = pd.read_csv(df_4h_file)
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], utc=True)

    print(f"✓ Loaded {len(df_4h):,} 4H bars")
    print(f"✓ Current columns: {len(df_4h.columns)}")

    # Resample features
    df_result = resample_features_to_4h(df_1h, df_4h)

    # Save
    output_file = project_root / 'data' / 'EURUSD_4H_2020_2025_with_macro.csv'
    print(f"\n📁 Saving to: {output_file}")
    df_result.to_csv(output_file, index=False)

    file_size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"✓ Saved ({file_size_mb:.2f} MB)")

    print("\n" + "="*80)
    print("✅ FEATURES ADDED")
    print("="*80)
    print(f"✓ Output: {output_file}")
    print(f"✓ Total bars: {len(df_result):,}")
    print(f"✓ Total columns: {len(df_result.columns)}")

    print("\n📊 Sample (first 5 rows, key columns):")
    sample_cols = ['timestamp', 'close', 'session_ny', 'sent_USD', 'spx_ret_1h', 'vix_level']
    available_cols = [c for c in sample_cols if c in df_result.columns]
    if available_cols:
        print(df_result[available_cols].head())


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
TASK 6: Internal Price Features Pipeline (4H Timeframe)
Compute price-derived technical features for 4H bars.

Parameter adjustments for 4H:
- Returns: 1, 3, 6, 12, 42 bars = 4h, 12h, 24h, 48h, 1week
- Volatility: 6, 18, 42 bars = 1day, 3days, 1week
- EMAs: 6, 18, 42 bars = 1day, 3days, 1week
- ATR: 14 bars (same relative period)
- RSI: 14 bars (same)
- Recent structure: 42 bars = 1 week
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def compute_log_returns(df: pd.DataFrame, periods: list) -> pd.DataFrame:
    """Compute log returns for multiple periods."""
    print("\nComputing log returns...")

    for period in periods:
        # Convert period to hours for naming
        hours = period * 4
        col_name = f'ret_{hours}h'
        df[col_name] = np.log(df['close'] / df['close'].shift(period))
        print(f"  ✓ {col_name} ({period} bars)")

    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute Average True Range (ATR)."""
    print(f"\nComputing ATR({period} bars)...")

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f'atr_{period}'] = true_range.rolling(window=period, min_periods=1).mean()

    print(f"  ✓ atr_{period}")
    return df


def compute_rolling_volatility(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Compute rolling volatility."""
    print("\nComputing rolling volatility...")

    if 'ret_4h' not in df.columns:
        df['ret_4h'] = np.log(df['close'] / df['close'].shift(1))

    for window in windows:
        hours = window * 4
        col_name = f'vol_{hours}h'
        df[col_name] = df['ret_4h'].rolling(window=window, min_periods=1).std()
        print(f"  ✓ {col_name} ({window} bars)")

    return df


def compute_ema(df: pd.DataFrame, periods: list) -> pd.DataFrame:
    """Compute Exponential Moving Averages."""
    print("\nComputing EMAs...")

    for period in periods:
        hours = period * 4
        col_name = f'ema_{hours}h'
        df[col_name] = df['close'].ewm(span=period, adjust=False).mean()
        print(f"  ✓ {col_name} ({period} bars)")

    return df


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute Relative Strength Index (RSI)."""
    print(f"\nComputing RSI({period} bars)...")

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    df[f'rsi_{period}'] = rsi
    print(f"  ✓ rsi_{period}")
    return df


def compute_candle_features(df: pd.DataFrame, atr_col: str = 'atr_14') -> pd.DataFrame:
    """Compute candle shape features normalized by ATR."""
    print("\nComputing candle shape features...")

    body = df['close'] - df['open']
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']

    df['body_norm'] = body / (df[atr_col] + 1e-10)
    df['upper_wick_norm'] = upper_wick / (df[atr_col] + 1e-10)
    df['lower_wick_norm'] = lower_wick / (df[atr_col] + 1e-10)

    print("  ✓ body_norm")
    print("  ✓ upper_wick_norm")
    print("  ✓ lower_wick_norm")

    return df


def compute_recent_structure(df: pd.DataFrame, window: int = 42) -> pd.DataFrame:
    """Compute distance from recent highs/lows."""
    print(f"\nComputing recent structure features (window={window} bars = 1 week)...")

    rolling_high = df['high'].rolling(window=window, min_periods=1).max()
    rolling_low = df['low'].rolling(window=window, min_periods=1).min()

    df[f'dist_high_{window}'] = (rolling_high - df['close']) / df['close']
    df[f'dist_low_{window}'] = (df['close'] - rolling_low) / df['close']

    df[f'bars_since_high_{window}'] = 0
    df[f'bars_since_low_{window}'] = 0

    for i in tqdm(range(window, len(df)), desc="Computing bars since extremes"):
        window_start = max(0, i - window)
        window_data = df.iloc[window_start:i+1]

        high_idx = window_data['high'].idxmax()
        low_idx = window_data['low'].idxmin()

        df.loc[df.index[i], f'bars_since_high_{window}'] = i - high_idx
        df.loc[df.index[i], f'bars_since_low_{window}'] = i - low_idx

    print(f"  ✓ dist_high_{window}")
    print(f"  ✓ dist_low_{window}")
    print(f"  ✓ bars_since_high_{window}")
    print(f"  ✓ bars_since_low_{window}")

    return df


def normalize_by_atr(df: pd.DataFrame, atr_col: str = 'atr_14') -> pd.DataFrame:
    """Normalize key features by ATR for regime-invariance."""
    print("\n" + "="*80)
    print("Normalizing Features by ATR (Regime-Invariant)")
    print("="*80)

    # Normalize returns
    return_features = ['ret_4h', 'ret_12h', 'ret_24h', 'ret_48h', 'ret_168h']
    for feat in return_features:
        if feat in df.columns:
            df[feat] = df[feat] / (df[atr_col] + 1e-8)
            print(f"  ✓ {feat} normalized by ATR")

    # Normalize EMA distances
    ema_features = ['ema_24h', 'ema_72h', 'ema_168h']
    for feat in ema_features:
        if feat in df.columns:
            df[feat] = (df['close'] - df[feat]) / (df[atr_col] + 1e-8)
            print(f"  ✓ {feat} converted to ATR-normalized distance")

    # Normalize volatility
    vol_features = ['vol_24h', 'vol_72h', 'vol_168h']
    for feat in vol_features:
        if feat in df.columns:
            df[feat] = df[feat] / (df[atr_col] + 1e-8)
            print(f"  ✓ {feat} normalized by ATR")

    print("\n✓ All features normalized by ATR for regime-invariance")
    return df


def compute_all_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all internal price features for 4H timeframe."""
    print("\n" + "="*80)
    print("Computing Internal Price Features (4H Timeframe)")
    print("="*80)

    df = df.copy()

    # 1. Log returns (1, 3, 6, 12, 42 bars = 4h, 12h, 24h, 48h, 1week)
    df = compute_log_returns(df, periods=[1, 3, 6, 12, 42])

    # 2. ATR(14 bars)
    df = compute_atr(df, period=14)

    # 3. Rolling volatility (6, 18, 42 bars = 1day, 3days, 1week)
    df = compute_rolling_volatility(df, windows=[6, 18, 42])

    # 4. EMAs (6, 18, 42 bars = 1day, 3days, 1week)
    df = compute_ema(df, periods=[6, 18, 42])

    # 5. RSI(14 bars)
    df = compute_rsi(df, period=14)

    # 6. Candle shape features
    df = compute_candle_features(df, atr_col='atr_14')

    # 7. Recent structure (42 bars = 1 week)
    df = compute_recent_structure(df, window=42)

    # 8. ATR-normalize key features
    df = normalize_by_atr(df, atr_col='atr_14')

    return df


def main():
    """Main pipeline execution"""

    print("="*80)
    print("TASK 6: Internal Price Features Pipeline (4H)")
    print("="*80)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Load 4H data with macro features
    input_file = project_root / 'data' / 'EURUSD_4H_2020_2025_with_macro.csv'
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    print(f"✓ Loaded {len(df):,} 4H bars")
    print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"✓ Current feature count: {len(df.columns)}")

    # Handle missing OHLC data
    print("\nHandling missing OHLC data...")
    missing_mask = df[['open', 'high', 'low', 'close']].isna().any(axis=1)
    missing_count = missing_mask.sum()
    print(f"  - Found {missing_count:,} bars with missing data ({missing_count/len(df)*100:.2f}%)")

    df['missing_flag'] = missing_mask.astype(int)
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill()
    print(f"  ✓ Forward-filled and added missing_flag")

    # Compute all price features
    df = compute_all_price_features(df)

    # Handle NaN values
    print("\n" + "="*80)
    print("Handling NaN values...")
    print("="*80)

    price_feature_cols = [
        col for col in df.columns
        if any(x in col for x in ['ret_', 'atr_', 'vol_', 'ema_', 'rsi_',
                                   'body_', 'wick_', 'dist_', 'bars_since'])
    ]

    print(f"Price feature columns: {len(price_feature_cols)}")

    nan_counts = df[price_feature_cols].isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts) > 0:
        print("\nNaN counts before filling:")
        for col, count in nan_counts.items():
            print(f"  - {col}: {count}")

    for col in price_feature_cols:
        if any(x in col for x in ['ret_', 'body_', 'wick_', 'dist_']):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].ffill().bfill()

    print("\n✓ NaN values handled")

    # Save
    output_file = project_root / 'data' / 'EURUSD_4H_2020_2025_with_price_features.csv'
    df.to_csv(output_file, index=False)

    print("\n" + "="*80)
    print("✅ TASK 6 COMPLETE (4H)")
    print("="*80)
    print(f"✓ Output: {output_file}")
    print(f"✓ Total bars: {len(df):,}")
    print(f"✓ Total features: {len(df.columns)}")

    print(f"\n✓ New price features ({len(price_feature_cols)}):")
    for col in price_feature_cols:
        print(f"  - {col}")

    print("\n📊 Sample (first 10 rows):")
    sample_cols = ['timestamp', 'close'] + price_feature_cols[:8]
    print(df[sample_cols].head(10))


if __name__ == "__main__":
    main()


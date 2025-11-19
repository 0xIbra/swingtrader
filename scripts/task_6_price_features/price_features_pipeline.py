#!/usr/bin/env python3
"""
TASK 6: Internal Price Features Pipeline
Compute price-derived technical features for ML modeling.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def compute_log_returns(df: pd.DataFrame, periods: list) -> pd.DataFrame:
    """
    Compute log returns for multiple periods.

    Args:
        df: DataFrame with 'close' column
        periods: List of periods (e.g., [1, 3, 6, 12, 24])

    Returns:
        DataFrame with return columns added
    """
    print("\nComputing log returns...")

    for period in periods:
        col_name = f'ret_{period}h'
        df[col_name] = np.log(df['close'] / df['close'].shift(period))
        print(f"  ✓ {col_name}")

    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute Average True Range (ATR).

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period (default 14)

    Returns:
        DataFrame with ATR column added
    """
    print(f"\nComputing ATR({period})...")

    # True Range components
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))

    # True Range is the maximum of the three
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # ATR is the rolling mean of True Range
    df[f'atr_{period}'] = true_range.rolling(window=period, min_periods=1).mean()

    print(f"  ✓ atr_{period}")

    return df


def compute_rolling_volatility(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """
    Compute rolling volatility (standard deviation of returns).

    Args:
        df: DataFrame with 'close' column
        windows: List of window sizes (e.g., [24, 72, 168])

    Returns:
        DataFrame with volatility columns added
    """
    print("\nComputing rolling volatility...")

    # First compute 1h returns if not already present
    if 'ret_1h' not in df.columns:
        df['ret_1h'] = np.log(df['close'] / df['close'].shift(1))

    for window in windows:
        col_name = f'vol_{window}h'
        df[col_name] = df['ret_1h'].rolling(window=window, min_periods=1).std()
        print(f"  ✓ {col_name}")

    return df


def compute_ema(df: pd.DataFrame, periods: list) -> pd.DataFrame:
    """
    Compute Exponential Moving Averages.

    Args:
        df: DataFrame with 'close' column
        periods: List of EMA periods (e.g., [24, 72, 168])

    Returns:
        DataFrame with EMA columns added
    """
    print("\nComputing EMAs...")

    for period in periods:
        col_name = f'ema_{period}'
        df[col_name] = df['close'].ewm(span=period, adjust=False).mean()
        print(f"  ✓ {col_name}")

    return df


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute Relative Strength Index (RSI).

    Args:
        df: DataFrame with 'close' column
        period: RSI period (default 14)

    Returns:
        DataFrame with RSI column added
    """
    print(f"\nComputing RSI({period})...")

    # Calculate price changes
    delta = df['close'].diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    # Calculate RS and RSI
    rs = avg_gain / (avg_loss + 1e-10)  # Add small value to avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    df[f'rsi_{period}'] = rsi

    print(f"  ✓ rsi_{period}")

    return df


def compute_candle_features(df: pd.DataFrame, atr_col: str = 'atr_14') -> pd.DataFrame:
    """
    Compute candle shape features normalized by ATR.

    Args:
        df: DataFrame with OHLC columns and ATR
        atr_col: Name of ATR column to use for normalization

    Returns:
        DataFrame with candle feature columns added
    """
    print("\nComputing candle shape features...")

    # Candle body (close - open)
    body = df['close'] - df['open']

    # Upper wick (high - max(open, close))
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)

    # Lower wick (min(open, close) - low)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']

    # Normalize by ATR
    df['body_norm'] = body / (df[atr_col] + 1e-10)
    df['upper_wick_norm'] = upper_wick / (df[atr_col] + 1e-10)
    df['lower_wick_norm'] = lower_wick / (df[atr_col] + 1e-10)

    print("  ✓ body_norm")
    print("  ✓ upper_wick_norm")
    print("  ✓ lower_wick_norm")

    return df


def compute_recent_structure(df: pd.DataFrame, window: int = 168) -> pd.DataFrame:
    """
    Compute distance from recent highs/lows and bars since extreme.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        window: Lookback window (default 168 = 1 week)

    Returns:
        DataFrame with structure feature columns added
    """
    print(f"\nComputing recent structure features (window={window})...")

    # Rolling high and low
    rolling_high = df['high'].rolling(window=window, min_periods=1).max()
    rolling_low = df['low'].rolling(window=window, min_periods=1).min()

    # Distance from high/low (normalized)
    df[f'dist_high_{window}'] = (rolling_high - df['close']) / df['close']
    df[f'dist_low_{window}'] = (df['close'] - rolling_low) / df['close']

    # Bars since high/low
    df[f'bars_since_high_{window}'] = 0
    df[f'bars_since_low_{window}'] = 0

    # Compute bars since high/low
    for i in tqdm(range(window, len(df)), desc="Computing bars since extremes"):
        # Get window
        window_start = max(0, i - window)
        window_data = df.iloc[window_start:i+1]

        # Find index of high and low in window
        high_idx = window_data['high'].idxmax()
        low_idx = window_data['low'].idxmin()

        # Calculate bars since
        df.loc[df.index[i], f'bars_since_high_{window}'] = i - high_idx
        df.loc[df.index[i], f'bars_since_low_{window}'] = i - low_idx

    print(f"  ✓ dist_high_{window}")
    print(f"  ✓ dist_low_{window}")
    print(f"  ✓ bars_since_high_{window}")
    print(f"  ✓ bars_since_low_{window}")

    return df


def normalize_by_atr(df: pd.DataFrame, atr_col: str = 'atr_14') -> pd.DataFrame:
    """
    Normalize key price features by ATR to make them regime-invariant.

    This helps the model generalize across different volatility regimes
    (e.g., calm vs volatile markets).

    Args:
        df: DataFrame with price features and ATR
        atr_col: Name of the ATR column to use for normalization

    Returns:
        DataFrame with ATR-normalized features added
    """
    print("\n" + "="*80)
    print("Normalizing Features by ATR (Regime-Invariant)")
    print("="*80)

    # Replace original features with ATR-normalized versions
    # This makes the model robust to volatility regime changes

    # 1. Normalize log returns by ATR
    return_features = ['ret_1h', 'ret_3h', 'ret_6h', 'ret_12h', 'ret_24h']
    for feat in return_features:
        if feat in df.columns:
            df[feat] = df[feat] / (df[atr_col] + 1e-8)  # Avoid division by zero
            print(f"  ✓ {feat} normalized by ATR")

    # 2. Normalize EMA distances by ATR
    ema_features = ['ema_24', 'ema_72', 'ema_168']
    for feat in ema_features:
        if feat in df.columns:
            # Convert EMA level to distance from current price, normalized by ATR
            df[feat] = (df['close'] - df[feat]) / (df[atr_col] + 1e-8)
            print(f"  ✓ {feat} converted to ATR-normalized distance from price")

    # 3. Normalize rolling volatility by ATR
    vol_features = ['vol_24h', 'vol_72h', 'vol_168h']
    for feat in vol_features:
        if feat in df.columns:
            df[feat] = df[feat] / (df[atr_col] + 1e-8)
            print(f"  ✓ {feat} normalized by ATR")

    # 4. Normalize recent structure distances by ATR (already done in compute_recent_structure)
    # dist_high_168 and dist_low_168 are already normalized

    print("\n✓ All key features normalized by ATR for regime-invariance")
    print("  Model will now generalize better across calm/volatile periods")

    return df


def compute_all_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all internal price features.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with all price features added
    """
    print("\n" + "="*80)
    print("Computing Internal Price Features")
    print("="*80)

    # Make a copy to avoid modifying original
    df = df.copy()

    # 1. Log returns (1, 3, 6, 12, 24 hours)
    df = compute_log_returns(df, periods=[1, 3, 6, 12, 24])

    # 2. ATR(14)
    df = compute_atr(df, period=14)

    # 3. Rolling volatility (1 day, 3 days, 1 week)
    df = compute_rolling_volatility(df, windows=[24, 72, 168])

    # 4. EMAs (1 day, 3 days, 1 week)
    df = compute_ema(df, periods=[24, 72, 168])

    # 5. RSI(14)
    df = compute_rsi(df, period=14)

    # 6. Candle shape features
    df = compute_candle_features(df, atr_col='atr_14')

    # 7. Recent structure (1 week = 168 hours)
    df = compute_recent_structure(df, window=168)

    # 8. ATR-normalize key features for regime invariance
    df = normalize_by_atr(df, atr_col='atr_14')

    return df


def main():
    """Main pipeline execution"""

    print("="*80)
    print("TASK 6: Internal Price Features Pipeline")
    print("="*80)

    # Get project root directory (cross-platform)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Load existing data with macro features
    input_file = project_root / 'data' / 'EURUSD_1H_2020_2025_with_macro.csv'
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    print(f"✓ Loaded {len(df):,} price bars")
    print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"✓ Current feature count: {len(df.columns)}")

    # Handle missing OHLC data (forex market gaps)
    print("\nHandling missing OHLC data (market gaps)...")
    missing_mask = df[['open', 'high', 'low', 'close']].isna().any(axis=1)
    missing_count = missing_mask.sum()
    print(f"  - Found {missing_count:,} bars with missing OHLC data ({missing_count/len(df)*100:.2f}%)")

    # Add missing flag before forward-filling
    df['missing_flag'] = missing_mask.astype(int)

    # Forward-fill OHLC data
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill()

    print(f"  ✓ Forward-filled OHLC data and added 'missing_flag' column")

    # Compute all price features
    df = compute_all_price_features(df)

    # Fill NaN values at the beginning (from lagged features)
    print("\n" + "="*80)
    print("Handling NaN values...")
    print("="*80)

    # Get price feature columns (excluding original OHLCV and existing features)
    price_feature_cols = [
        col for col in df.columns
        if any(x in col for x in ['ret_', 'atr_', 'vol_', 'ema_', 'rsi_',
                                   'body_', 'wick_', 'dist_', 'bars_since'])
    ]

    print(f"Price feature columns: {len(price_feature_cols)}")

    # Check NaN counts before filling
    print("\nNaN counts before filling:")
    nan_counts = df[price_feature_cols].isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts) > 0:
        for col, count in nan_counts.items():
            print(f"  - {col}: {count}")
    else:
        print("  No NaN values found")

    # Fill NaN values
    # For returns and normalized features: fill with 0
    # For levels (EMA, ATR, volatility): forward fill then backfill
    for col in price_feature_cols:
        if any(x in col for x in ['ret_', 'body_', 'wick_', 'dist_']):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].ffill().bfill()

    print("\n✓ NaN values handled")

    # Save enhanced data
    output_file = project_root / 'data' / 'EURUSD_1H_2020_2025_with_price_features.csv'
    df.to_csv(output_file, index=False)

    print("\n" + "="*80)
    print("✅ TASK 6 COMPLETE")
    print("="*80)
    print(f"✓ Output file: {output_file}")
    print(f"✓ Total rows: {len(df):,}")
    print(f"✓ Total features: {len(df.columns)}")

    print(f"\n✓ All feature columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        marker = "🆕" if col in price_feature_cols else "  "
        print(f"  {marker} {i:2d}. {col}")

    print(f"\n📊 New price features added ({len(price_feature_cols)}):")
    for col in price_feature_cols:
        print(f"  - {col}")

    print("\n📊 Sample of price features (first 10 rows):")
    sample_cols = ['timestamp', 'close'] + price_feature_cols[:8]
    print(df[sample_cols].head(10))

    print("\n📊 Price feature statistics:")
    print(df[price_feature_cols].describe())


if __name__ == "__main__":
    main()


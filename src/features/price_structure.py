"""
Price structure analysis: support/resistance levels, pattern detection.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.signal import argrelextrema
import config


class PriceStructureAnalyzer:
    """Detects support/resistance levels and chart patterns."""

    def __init__(self, tolerance: float = None):
        """
        Initialize analyzer.

        Args:
            tolerance: Price tolerance for level clustering (defaults to config)
        """
        self.tolerance = tolerance or config.SUPPORT_RESISTANCE_TOLERANCE

    def find_swing_points(self, df: pd.DataFrame, order: int = 5) -> Tuple[pd.Series, pd.Series]:
        """
        Find swing highs and lows in price data.

        Args:
            df: DataFrame with OHLCV data
            order: Number of candles on each side to compare

        Returns:
            Tuple of (swing_highs, swing_lows) as boolean Series
        """
        # Find local maxima (swing highs)
        high_indices = argrelextrema(df['high'].values, np.greater, order=order)[0]
        swing_highs = pd.Series(False, index=df.index)
        swing_highs.iloc[high_indices] = True

        # Find local minima (swing lows)
        low_indices = argrelextrema(df['low'].values, np.less, order=order)[0]
        swing_lows = pd.Series(False, index=df.index)
        swing_lows.iloc[low_indices] = True

        return swing_highs, swing_lows

    def cluster_levels(self, prices: List[float]) -> List[Dict]:
        """
        Cluster price levels that are close together.

        Args:
            prices: List of price levels

        Returns:
            List of dictionaries with 'level' and 'touches'
        """
        if not prices:
            return []

        prices = sorted(prices)
        clusters = []
        current_cluster = [prices[0]]

        for price in prices[1:]:
            # If price is within tolerance of cluster mean, add to cluster
            cluster_mean = np.mean(current_cluster)
            if abs(price - cluster_mean) / cluster_mean <= self.tolerance:
                current_cluster.append(price)
            else:
                # Save current cluster and start new one
                clusters.append({
                    'level': np.mean(current_cluster),
                    'touches': len(current_cluster)
                })
                current_cluster = [price]

        # Add the last cluster
        if current_cluster:
            clusters.append({
                'level': np.mean(current_cluster),
                'touches': len(current_cluster)
            })

        # Sort by number of touches (strongest first)
        clusters.sort(key=lambda x: x['touches'], reverse=True)

        return clusters

    def detect_support_resistance(self, df: pd.DataFrame,
                                  lookback: int = None) -> Dict:
        """
        Detect support and resistance levels.

        Args:
            df: DataFrame with OHLCV data
            lookback: Number of bars to analyze (defaults to config)

        Returns:
            Dictionary with support and resistance levels
        """
        lookback = lookback or config.LOOKBACK_BARS
        df_recent = df.iloc[-lookback:]

        # Find swing points
        swing_highs, swing_lows = self.find_swing_points(df_recent)

        # Get swing high and low prices
        high_prices = df_recent.loc[swing_highs, 'high'].tolist()
        low_prices = df_recent.loc[swing_lows, 'low'].tolist()

        # Cluster into levels
        resistance_levels = self.cluster_levels(high_prices)
        support_levels = self.cluster_levels(low_prices)

        return {
            'resistance': resistance_levels,
            'support': support_levels
        }

    def get_nearest_levels(self, df: pd.DataFrame, current_price: float = None) -> Dict:
        """
        Get nearest support and resistance levels to current price.

        Args:
            df: DataFrame with OHLCV data
            current_price: Current price (defaults to last close)

        Returns:
            Dictionary with nearest support/resistance info
        """
        if current_price is None:
            current_price = df['close'].iloc[-1]

        levels = self.detect_support_resistance(df)

        # Find nearest support (below current price)
        nearest_support = None
        support_distance = float('inf')
        support_strength = 0

        for level in levels['support']:
            if level['level'] < current_price:
                distance = current_price - level['level']
                if distance < support_distance:
                    support_distance = distance
                    nearest_support = level['level']
                    support_strength = level['touches']

        # Find nearest resistance (above current price)
        nearest_resistance = None
        resistance_distance = float('inf')
        resistance_strength = 0

        for level in levels['resistance']:
            if level['level'] > current_price:
                distance = level['level'] - current_price
                if distance < resistance_distance:
                    resistance_distance = distance
                    nearest_resistance = level['level']
                    resistance_strength = level['touches']

        # Calculate distances as percentage
        support_distance_pct = (support_distance / current_price * 100) if nearest_support else None
        resistance_distance_pct = (resistance_distance / current_price * 100) if nearest_resistance else None

        return {
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_strength': support_strength,
            'resistance_strength': resistance_strength,
            'distance_to_support_pct': support_distance_pct,
            'distance_to_resistance_pct': resistance_distance_pct
        }

    def detect_double_bottom(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect double bottom pattern.

        A double bottom consists of:
        - Two lows at similar price (within tolerance)
        - Separated by a high in between
        - Second low has lower/similar volume
        - Currently breaking above the middle high

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Pattern info dict if detected, None otherwise
        """
        if len(df) < 20:
            return None

        # Look at recent data
        recent = df.iloc[-50:]
        swing_highs, swing_lows = self.find_swing_points(recent, order=3)

        # Get last few swing lows
        low_indices = recent[swing_lows].index
        if len(low_indices) < 2:
            return None

        # Check last two lows
        low1_idx = low_indices[-2]
        low2_idx = low_indices[-1]

        low1_price = recent.loc[low1_idx, 'low']
        low2_price = recent.loc[low2_idx, 'low']

        # Check if lows are at similar price
        price_diff = abs(low1_price - low2_price) / low1_price
        if price_diff > self.tolerance:
            return None

        # Check if there's a high in between
        between = recent.loc[low1_idx:low2_idx]
        if len(between) < 3:
            return None

        middle_high = between['high'].max()

        # Check if middle high is above both lows
        if middle_high <= max(low1_price, low2_price):
            return None

        # Check volume (second low should have lower/similar volume)
        low1_vol = recent.loc[low1_idx, 'volume']
        low2_vol = recent.loc[low2_idx, 'volume']
        volume_declining = low2_vol <= low1_vol * 1.2

        # Check if price is breaking above middle high
        current_price = recent['close'].iloc[-1]
        breakout = current_price > middle_high

        return {
            'pattern': 'double_bottom',
            'low1': low1_price,
            'low2': low2_price,
            'middle_high': middle_high,
            'volume_declining': volume_declining,
            'breakout': breakout,
            'confirmed': breakout and volume_declining
        }

    def detect_double_top(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect double top pattern.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Pattern info dict if detected, None otherwise
        """
        if len(df) < 20:
            return None

        recent = df.iloc[-50:]
        swing_highs, swing_lows = self.find_swing_points(recent, order=3)

        # Get last few swing highs
        high_indices = recent[swing_highs].index
        if len(high_indices) < 2:
            return None

        # Check last two highs
        high1_idx = high_indices[-2]
        high2_idx = high_indices[-1]

        high1_price = recent.loc[high1_idx, 'high']
        high2_price = recent.loc[high2_idx, 'high']

        # Check if highs are at similar price
        price_diff = abs(high1_price - high2_price) / high1_price
        if price_diff > self.tolerance:
            return None

        # Check if there's a low in between
        between = recent.loc[high1_idx:high2_idx]
        if len(between) < 3:
            return None

        middle_low = between['low'].min()

        # Check if middle low is below both highs
        if middle_low >= min(high1_price, high2_price):
            return None

        # Check volume
        high1_vol = recent.loc[high1_idx, 'volume']
        high2_vol = recent.loc[high2_idx, 'volume']
        volume_declining = high2_vol <= high1_vol * 1.2

        # Check if price is breaking below middle low
        current_price = recent['close'].iloc[-1]
        breakout = current_price < middle_low

        return {
            'pattern': 'double_top',
            'high1': high1_price,
            'high2': high2_price,
            'middle_low': middle_low,
            'volume_declining': volume_declining,
            'breakout': breakout,
            'confirmed': breakout and volume_declining
        }

    def detect_bull_flag(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect bull flag pattern.

        Bull flag:
        - Strong upward move (pole)
        - Consolidation with slight downward drift (flag)
        - Lower volume during consolidation
        - Breakout above flag resistance

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Pattern info dict if detected, None otherwise
        """
        if len(df) < 30:
            return None

        recent = df.iloc[-30:]

        # Check for strong upward move (pole)
        # Look at first 10 bars
        pole = recent.iloc[:10]
        pole_gain = (pole['close'].iloc[-1] - pole['close'].iloc[0]) / pole['close'].iloc[0]

        if pole_gain < 0.015:  # At least 1.5% gain
            return None

        # Check for consolidation (flag)
        flag = recent.iloc[10:]
        flag_range = (flag['high'].max() - flag['low'].min()) / flag['close'].mean()

        if flag_range > 0.01:  # Flag should be tight, less than 1%
            return None

        # Check volume declining during flag
        pole_avg_vol = pole['volume'].mean()
        flag_avg_vol = flag['volume'].mean()
        volume_declining = flag_avg_vol < pole_avg_vol * 0.8

        # Check if breaking out
        flag_resistance = flag['high'].max()
        current_price = recent['close'].iloc[-1]
        breakout = current_price > flag_resistance

        return {
            'pattern': 'bull_flag',
            'pole_gain': pole_gain,
            'flag_resistance': flag_resistance,
            'volume_declining': volume_declining,
            'breakout': breakout,
            'confirmed': breakout and volume_declining
        }

    def detect_pattern(self, df: pd.DataFrame) -> str:
        """
        Detect the most prominent pattern in the data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Pattern name or 'none'
        """
        # Check for each pattern
        double_bottom = self.detect_double_bottom(df)
        if double_bottom and double_bottom['confirmed']:
            return 'double_bottom'

        double_top = self.detect_double_top(df)
        if double_top and double_top['confirmed']:
            return 'double_top'

        bull_flag = self.detect_bull_flag(df)
        if bull_flag and bull_flag['confirmed']:
            return 'bull_flag'

        return 'none'


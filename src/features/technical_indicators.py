"""
Technical indicators using TA-Lib.
"""
import pandas as pd
import numpy as np
from typing import Dict


class TechnicalIndicators:
    """Calculates technical indicators for trading signals."""

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()

        # RSI
        df['rsi_14'] = TechnicalIndicators.calculate_rsi(df['close'], period=14)

        # ATR
        df['atr'] = TechnicalIndicators.calculate_atr(df, period=14)

        # MACD
        macd_data = TechnicalIndicators.calculate_macd(df['close'])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']

        # EMAs for trend
        df['ema_20'] = TechnicalIndicators.calculate_ema(df['close'], period=20)
        df['ema_50'] = TechnicalIndicators.calculate_ema(df['close'], period=50)

        # Bollinger Bands
        bb_data = TechnicalIndicators.calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_data['upper']
        df['bb_middle'] = bb_data['middle']
        df['bb_lower'] = bb_data['lower']

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # ROC (Rate of Change / Momentum)
        df['momentum_20'] = TechnicalIndicators.calculate_roc(df['close'], period=20)

        return df

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index).

        Args:
            prices: Series of prices
            period: RSI period

        Returns:
            RSI values
        """
        delta = prices.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate ATR (Average True Range).

        Args:
            df: DataFrame with high, low, close
            period: ATR period

        Returns:
            ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12,
                       slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            prices: Series of prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            Dictionary with 'macd', 'signal', 'histogram'
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line

        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate EMA (Exponential Moving Average).

        Args:
            prices: Series of prices
            period: EMA period

        Returns:
            EMA values
        """
        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate SMA (Simple Moving Average).

        Args:
            prices: Series of prices
            period: SMA period

        Returns:
            SMA values
        """
        return prices.rolling(window=period).mean()

    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20,
                                  std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Series of prices
            period: Moving average period
            std_dev: Number of standard deviations

        Returns:
            Dictionary with 'upper', 'middle', 'lower' bands
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }

    @staticmethod
    def calculate_roc(prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate ROC (Rate of Change).

        Args:
            prices: Series of prices
            period: ROC period

        Returns:
            ROC values as percentage
        """
        roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
        return roc

    @staticmethod
    def detect_trend(df: pd.DataFrame, ema_short: int = 20,
                     ema_long: int = 50) -> int:
        """
        Detect trend direction using EMA crossover.

        Args:
            df: DataFrame with price data and EMAs
            ema_short: Short EMA period
            ema_long: Long EMA period

        Returns:
            1 for uptrend, -1 for downtrend, 0 for sideways
        """
        if f'ema_{ema_short}' not in df.columns:
            df[f'ema_{ema_short}'] = TechnicalIndicators.calculate_ema(df['close'], ema_short)
        if f'ema_{ema_long}' not in df.columns:
            df[f'ema_{ema_long}'] = TechnicalIndicators.calculate_ema(df['close'], ema_long)

        ema_short_val = df[f'ema_{ema_short}'].iloc[-1]
        ema_long_val = df[f'ema_{ema_long}'].iloc[-1]

        # Check if there's a clear trend
        diff_pct = abs(ema_short_val - ema_long_val) / ema_long_val

        if diff_pct < 0.001:  # Less than 0.1% difference
            return 0  # Sideways
        elif ema_short_val > ema_long_val:
            return 1  # Uptrend
        else:
            return -1  # Downtrend

    @staticmethod
    def calculate_volume_trend(df: pd.DataFrame, period: int = 10) -> int:
        """
        Calculate if volume is increasing or decreasing.

        Args:
            df: DataFrame with volume data
            period: Period to analyze

        Returns:
            1 for increasing, -1 for decreasing, 0 for neutral
        """
        recent_vol = df['volume'].iloc[-period:]

        # Simple linear regression slope
        x = np.arange(len(recent_vol))
        y = recent_vol.values

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        avg_vol = y.mean()

        # Normalize slope by average volume
        normalized_slope = slope / avg_vol

        if normalized_slope > 0.05:
            return 1  # Increasing
        elif normalized_slope < -0.05:
            return -1  # Decreasing
        else:
            return 0  # Neutral


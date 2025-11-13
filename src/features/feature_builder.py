"""
Feature builder that combines all feature engineering components.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from .price_structure import PriceStructureAnalyzer
from .technical_indicators import TechnicalIndicators
from .sentiment_analyzer import SentimentAnalyzer
import logging

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Builds complete feature set for ML models."""

    def __init__(self):
        """Initialize feature builder with all analyzers."""
        self.price_analyzer = PriceStructureAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()

    def build_features(self, df: pd.DataFrame, instrument: str,
                      headlines: Optional[list] = None,
                      include_patterns: bool = True) -> pd.DataFrame:
        """
        Build complete feature set from price data and news.

        Args:
            df: DataFrame with OHLCV data
            instrument: Currency pair
            headlines: List of news headlines
            include_patterns: Whether to detect patterns (slower)

        Returns:
            DataFrame with all features
        """
        # Make a copy to avoid modifying original
        features_df = df.copy()

        # Add technical indicators
        features_df = TechnicalIndicators.add_all_indicators(features_df)

        # Detect trends for multiple timeframes (requires having the data)
        features_df['trend'] = features_df.apply(
            lambda row: TechnicalIndicators.detect_trend(features_df.loc[:row.name]),
            axis=1
        )

        # Calculate volume trend
        features_df['volume_trend'] = TechnicalIndicators.calculate_volume_trend(features_df)

        # Get support/resistance levels for each row
        for idx in range(len(features_df)):
            if idx < 50:  # Need enough history
                features_df.loc[features_df.index[idx], 'nearest_support'] = np.nan
                features_df.loc[features_df.index[idx], 'nearest_resistance'] = np.nan
                features_df.loc[features_df.index[idx], 'support_strength'] = 0
                features_df.loc[features_df.index[idx], 'resistance_strength'] = 0
                features_df.loc[features_df.index[idx], 'distance_to_support_pct'] = np.nan
                features_df.loc[features_df.index[idx], 'distance_to_resistance_pct'] = np.nan
            else:
                df_subset = features_df.iloc[:idx+1]
                levels = self.price_analyzer.get_nearest_levels(df_subset)

                features_df.loc[features_df.index[idx], 'nearest_support'] = levels['nearest_support']
                features_df.loc[features_df.index[idx], 'nearest_resistance'] = levels['nearest_resistance']
                features_df.loc[features_df.index[idx], 'support_strength'] = levels['support_strength']
                features_df.loc[features_df.index[idx], 'resistance_strength'] = levels['resistance_strength']
                features_df.loc[features_df.index[idx], 'distance_to_support_pct'] = levels['distance_to_support_pct']
                features_df.loc[features_df.index[idx], 'distance_to_resistance_pct'] = levels['distance_to_resistance_pct']

        # Add pattern detection if requested (only for last row, as it's expensive)
        if include_patterns:
            pattern = self.price_analyzer.detect_pattern(features_df)
            features_df['pattern_type'] = pattern

        return features_df

    def extract_current_features(self, df: pd.DataFrame, instrument: str,
                                headlines: Optional[list] = None,
                                market_regime: str = 'neutral',
                                high_impact_event_24h: bool = False) -> Dict:
        """
        Extract feature vector for the current bar (for real-time prediction).

        Args:
            df: DataFrame with OHLCV data (with indicators already calculated)
            instrument: Currency pair
            headlines: Recent news headlines
            market_regime: Current market regime
            high_impact_event_24h: Whether high-impact event is expected

        Returns:
            Dictionary with feature values
        """
        current = df.iloc[-1]

        # Get support/resistance
        levels = self.price_analyzer.get_nearest_levels(df)

        # Get pattern
        pattern = self.price_analyzer.detect_pattern(df)

        # Encode pattern type
        pattern_encoding = {
            'none': 0,
            'double_bottom': 1,
            'double_top': 2,
            'bull_flag': 3
        }

        # Get sentiment if headlines provided
        news_sentiment = 0.0
        if headlines:
            news_sentiment = self.sentiment_analyzer.analyze_instrument_sentiment(
                headlines, instrument
            )

        # Market regime encoding
        regime_encoding = {
            'risk_off': 0,
            'neutral': 1,
            'risk_on': 2
        }

        # Build feature dictionary (20 features as per specs)
        features = {
            # Price structure (4 features)
            'support_level_strength': levels['support_strength'],
            'resistance_level_strength': levels['resistance_strength'],
            'distance_to_support_pct': levels['distance_to_support_pct'] or 0,
            'distance_to_resistance_pct': levels['distance_to_resistance_pct'] or 0,

            # Price volatility (1 feature)
            'price_volatility': current['atr'] / current['close'],  # Normalized ATR

            # Momentum (3 features)
            'rsi_14': current['rsi_14'],
            'momentum_20': current['momentum_20'],
            'macd_histogram': current['macd_histogram'],

            # Volume (2 features)
            'volume_ratio': current['volume_ratio'],
            'volume_trend': TechnicalIndicators.calculate_volume_trend(df.iloc[-10:]),

            # Multi-timeframe trend (would need data from different timeframes)
            # For now, using single timeframe
            'trend_1h': TechnicalIndicators.detect_trend(df),
            'trend_4h': TechnicalIndicators.detect_trend(df),  # Would use 4H data
            'trend_daily': TechnicalIndicators.detect_trend(df),  # Would use daily data
            'timeframe_alignment_score': 1.0,  # Placeholder, would calculate from multi-TF

            # Pattern (1 feature)
            'pattern_type': pattern_encoding.get(pattern, 0),

            # Context (5 features)
            'news_sentiment': news_sentiment,
            'news_urgency': len(headlines) / 20 if headlines else 0,  # Normalized
            'market_regime': regime_encoding.get(market_regime, 1),
            'high_impact_event_24h': 1 if high_impact_event_24h else 0,
            'session_overlap': 1  # Placeholder, would check actual time
        }

        return features

    def build_training_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build features optimized for training (faster, batch processing).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with features suitable for training
        """
        # Add all technical indicators
        df = TechnicalIndicators.add_all_indicators(df)

        # Calculate price volatility
        df['price_volatility'] = df['atr'] / df['close']

        # Volume trend (vectorized for speed)
        df['volume_trend'] = 0

        # Simple trend detection
        df['trend'] = 0
        df.loc[df['ema_20'] > df['ema_50'], 'trend'] = 1
        df.loc[df['ema_20'] < df['ema_50'], 'trend'] = -1

        return df


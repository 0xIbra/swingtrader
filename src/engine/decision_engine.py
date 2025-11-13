"""
Decision engine - combines all signals and decides whether to trade.
"""
import pandas as pd
from typing import Dict, Optional, Tuple
from ..models.bounce_predictor import BouncePredictor
from ..models.direction_predictor import DirectionPredictor
from ..data.database import TradeDatabase
from ..data.oanda_client import OANDAClient
from ..data.news_scraper import NewsScraper
from ..features.feature_builder import FeatureBuilder
import config
import logging

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Makes trading decisions based on multiple signals and filters."""

    def __init__(self, bounce_model_path: str = None, direction_model_path: str = None):
        """
        Initialize decision engine.

        Args:
            bounce_model_path: Path to trained bounce model
            direction_model_path: Path to trained direction model
        """
        # Load models
        bounce_path = bounce_model_path or str(config.MODELS_DIR / "bounce_predictor.joblib")
        direction_path = direction_model_path or str(config.MODELS_DIR / "direction_predictor.joblib")

        self.bounce_predictor = BouncePredictor(bounce_path)
        self.direction_predictor = DirectionPredictor(direction_path)

        # Initialize components
        self.database = TradeDatabase()
        self.news_scraper = NewsScraper()
        self.feature_builder = FeatureBuilder()
        self.oanda_client = OANDAClient()

    def evaluate_setup(self, instrument: str, df: pd.DataFrame,
                       headlines: Optional[list] = None) -> Dict:
        """
        Evaluate a potential trading setup through multi-stage filtering.

        Args:
            instrument: Currency pair
            df: DataFrame with OHLCV data and features
            headlines: Recent news headlines

        Returns:
            Dictionary with evaluation results and decision
        """
        result = {
            'instrument': instrument,
            'decision': 'SKIP',
            'reason': None,
            'confidence': 0.0,
            'direction': None,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'risk_percent': 0.0
        }

        current = df.iloc[-1]
        current_price = current['close']
        atr = current['atr']

        # STEP 1: Check if we're near a level
        logger.info(f"Step 1: Checking price proximity to levels...")

        distance_to_support = current.get('distance_to_support_pct', float('inf'))
        distance_to_resistance = current.get('distance_to_resistance_pct', float('inf'))

        near_support = distance_to_support < 0.5
        near_resistance = distance_to_resistance < 0.5

        if not (near_support or near_resistance):
            result['reason'] = "Not near support or resistance level"
            logger.info(f"❌ {result['reason']}")
            return result

        logger.info(f"✓ Near level: support={near_support}, resistance={near_resistance}")

        # STEP 2: Pattern recognition - Bounce predictor
        logger.info(f"Step 2: Evaluating bounce probability...")

        # Extract features for bounce predictor
        bounce_features = {
            'support_level_strength': current.get('support_strength', 0),
            'resistance_level_strength': current.get('resistance_strength', 0),
            'distance_to_support_pct': distance_to_support if not pd.isna(distance_to_support) else 0,
            'distance_to_resistance_pct': distance_to_resistance if not pd.isna(distance_to_resistance) else 0,
            'price_volatility': current.get('price_volatility', 0),
            'rsi_14': current.get('rsi_14', 50),
            'momentum_20': current.get('momentum_20', 0),
            'macd_histogram': current.get('macd_histogram', 0),
            'volume_ratio': current.get('volume_ratio', 1),
            'volume_trend': current.get('volume_trend', 0),
            'trend': current.get('trend', 0)
        }

        bounce_prob = self.bounce_predictor.predict_single(bounce_features)

        if bounce_prob < config.BOUNCE_MODEL_THRESHOLD:
            result['reason'] = f"Bounce probability too low: {bounce_prob:.2f}"
            logger.info(f"❌ {result['reason']}")
            return result

        logger.info(f"✓ Bounce probability: {bounce_prob:.2f}")

        # STEP 3: Direction confirmation
        logger.info(f"Step 3: Checking direction probability...")

        direction_probs = self.direction_predictor.predict_single(bounce_features)

        # Determine expected direction based on level
        if near_support:
            expected_direction = 'UP'
            direction_prob = direction_probs['prob_up']
        else:  # near_resistance
            expected_direction = 'DOWN'
            direction_prob = direction_probs['prob_down']

        if direction_prob < config.DIRECTION_MODEL_THRESHOLD:
            result['reason'] = f"Direction probability too low: {direction_prob:.2f}"
            logger.info(f"❌ {result['reason']}")
            return result

        logger.info(f"✓ Direction probability ({expected_direction}): {direction_prob:.2f}")

        # STEP 4: Context check
        logger.info(f"Step 4: Checking market context...")

        # News sentiment
        news_sentiment = 0.0
        market_regime = 'neutral'
        high_impact_event = False

        if headlines:
            news_sentiment = self.feature_builder.sentiment_analyzer.analyze_instrument_sentiment(
                headlines, instrument
            )

            regime_indicators = self.news_scraper.get_market_regime_indicators()
            market_regime = regime_indicators['regime']

            high_impact_event = self.news_scraper.check_high_impact_events(hours_ahead=24)

        # Skip if high impact event coming
        if high_impact_event:
            result['reason'] = "High impact event within 24 hours"
            logger.info(f"❌ {result['reason']}")
            return result

        # Check sentiment alignment
        if expected_direction == 'UP' and news_sentiment < -0.3:
            result['reason'] = "News sentiment contradicts bullish setup"
            logger.info(f"❌ {result['reason']}")
            return result

        if expected_direction == 'DOWN' and news_sentiment > 0.3:
            result['reason'] = "News sentiment contradicts bearish setup"
            logger.info(f"❌ {result['reason']}")
            return result

        logger.info(f"✓ Context OK: sentiment={news_sentiment:.2f}, regime={market_regime}")

        # STEP 5: Timeframe alignment
        logger.info(f"Step 5: Checking timeframe alignment...")

        # For now, using single timeframe
        # In production, would fetch multiple timeframes
        trend = current.get('trend', 0)

        if expected_direction == 'UP' and trend == -1:
            result['reason'] = "Trend not aligned (downtrend)"
            logger.info(f"❌ {result['reason']}")
            return result

        if expected_direction == 'DOWN' and trend == 1:
            result['reason'] = "Trend not aligned (uptrend)"
            logger.info(f"❌ {result['reason']}")
            return result

        timeframe_alignment = 1.0 if abs(trend) == 1 else 0.66

        if timeframe_alignment < config.TIMEFRAME_ALIGNMENT_THRESHOLD:
            result['reason'] = f"Timeframe alignment too low: {timeframe_alignment:.2f}"
            logger.info(f"❌ {result['reason']}")
            return result

        logger.info(f"✓ Timeframe alignment: {timeframe_alignment:.2f}")

        # STEP 6: Experience lookup (confidence calculation)
        logger.info(f"Step 6: Calculating confidence from historical data...")

        pattern_type = self.feature_builder.price_analyzer.detect_pattern(df)

        confidence_data = self.database.calculate_confidence(
            pattern_type=pattern_type,
            market_regime=market_regime,
            timeframe_alignment=timeframe_alignment,
            current_sentiment=news_sentiment
        )

        confidence = confidence_data['confidence']

        if confidence < config.MIN_CONFIDENCE:
            result['reason'] = f"Confidence too low: {confidence:.2f}"
            logger.info(f"❌ {result['reason']}")
            return result

        logger.info(f"✓ Confidence: {confidence:.2f} (based on {confidence_data['sample_size']} similar trades)")

        # STEP 7: Calculate position details
        logger.info(f"Step 7: Calculating position parameters...")

        # Direction
        direction = 'LONG' if expected_direction == 'UP' else 'SHORT'

        # Entry price (current market price)
        entry_price = current_price

        # Stop loss (1.5 ATR from entry)
        if direction == 'LONG':
            stop_loss = entry_price - (config.STOP_LOSS_ATR_MULTIPLIER * atr)
            take_profit = entry_price + (config.TAKE_PROFIT_ATR_MULTIPLIER * atr)
        else:  # SHORT
            stop_loss = entry_price + (config.STOP_LOSS_ATR_MULTIPLIER * atr)
            take_profit = entry_price - (config.TAKE_PROFIT_ATR_MULTIPLIER * atr)

        # Risk percent (scaled by confidence)
        risk_percent = config.BASE_RISK_PERCENT + (confidence - 0.5) * (config.MAX_RISK_PERCENT - config.BASE_RISK_PERCENT)
        risk_percent = min(risk_percent, config.MAX_RISK_PERCENT)

        # All checks passed - TRADE
        result['decision'] = 'TRADE'
        result['direction'] = direction
        result['entry_price'] = entry_price
        result['stop_loss'] = stop_loss
        result['take_profit'] = take_profit
        result['confidence'] = confidence
        result['risk_percent'] = risk_percent
        result['reason'] = "All criteria met"

        result['details'] = {
            'bounce_prob': bounce_prob,
            'direction_prob': direction_prob,
            'news_sentiment': news_sentiment,
            'market_regime': market_regime,
            'pattern_type': pattern_type,
            'timeframe_alignment': timeframe_alignment,
            'atr': atr,
            'historical_sample_size': confidence_data['sample_size'],
            'historical_win_rate': confidence_data['win_rate']
        }

        logger.info(f"✅ TRADE SIGNAL: {direction} {instrument}")
        logger.info(f"   Entry: {entry_price:.5f}")
        logger.info(f"   Stop: {stop_loss:.5f}")
        logger.info(f"   Target: {take_profit:.5f}")
        logger.info(f"   Risk: {risk_percent:.2f}%")

        return result

    def scan_for_setups(self, instruments: Optional[list] = None) -> list:
        """
        Scan multiple instruments for trading setups.

        Args:
            instruments: List of currency pairs to scan (defaults to config)

        Returns:
            List of trade signals
        """
        if instruments is None:
            instruments = config.INSTRUMENTS

        signals = []

        for instrument in instruments:
            logger.info(f"\n{'='*60}")
            logger.info(f"Scanning {instrument}")
            logger.info(f"{'='*60}")

            try:
                # Fetch recent data
                df = self.oanda_client.get_candles(
                    instrument=instrument,
                    granularity="H4",
                    count=200
                )

                if df.empty:
                    logger.warning(f"No data for {instrument}")
                    continue

                # Build features
                df = self.feature_builder.build_training_features(df)

                # Get recent news
                headlines = self.news_scraper.fetch_instrument_specific_news(
                    instrument, hours_back=24
                )

                # Evaluate setup
                result = self.evaluate_setup(instrument, df, headlines)

                if result['decision'] == 'TRADE':
                    signals.append(result)

            except Exception as e:
                logger.error(f"Error scanning {instrument}: {e}")
                continue

        return signals


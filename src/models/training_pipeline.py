"""
Training pipeline for preparing data and training both models.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from ..data.data_provider_factory import get_data_client
from ..features.feature_builder import FeatureBuilder
from .bounce_predictor import BouncePredictor
from .direction_predictor import DirectionPredictor
import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates data collection, feature engineering, and model training."""

    def __init__(self):
        """Initialize training pipeline."""
        self.data_client = get_data_client()
        self.feature_builder = FeatureBuilder()
        self.bounce_model = BouncePredictor()
        self.direction_model = DirectionPredictor()

    def fetch_training_data(self, instrument: str, granularity: str = "H4",
                           days_back: int = 1825) -> pd.DataFrame:
        """
        Fetch historical data for training (5 years recommended).

        Args:
            instrument: Currency pair
            granularity: Timeframe
            days_back: Number of days of historical data

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {days_back} days of {granularity} data for {instrument}")

        df = self.data_client.get_historical_data(
            instrument=instrument,
            granularity=granularity,
            days_back=days_back
        )

        logger.info(f"Fetched {len(df)} bars")

        return df

    def prepare_features(self, df: pd.DataFrame, instrument: str) -> pd.DataFrame:
        """
        Prepare features from raw OHLCV data.

        Args:
            df: DataFrame with OHLCV data
            instrument: Currency pair

        Returns:
            DataFrame with features
        """
        logger.info("Building features...")

        # Use the faster training-optimized feature builder
        features_df = self.feature_builder.build_training_features(df)

        logger.info(f"Features built: {features_df.shape}")

        return features_df

    def train_bounce_model(self, df: pd.DataFrame,
                          save_path: Optional[str] = None) -> Dict:
        """
        Train bounce predictor model.

        Args:
            df: DataFrame with features
            save_path: Path to save trained model

        Returns:
            Training metrics
        """
        logger.info("Training bounce predictor...")

        # Prepare training data
        X, y = self.bounce_model.prepare_training_data(df)

        if len(X) < 100:
            raise ValueError(f"Not enough training samples: {len(X)}. Need at least 100.")

        # Train model
        metrics = self.bounce_model.train(X, y)

        # Save model
        if save_path is None:
            save_path = config.MODELS_DIR / "bounce_predictor.joblib"

        self.bounce_model.save_model(str(save_path))

        # Log feature importance
        importance = self.bounce_model.get_feature_importance()
        logger.info("Top 5 features:")
        logger.info(importance.head())

        return metrics

    def train_direction_model(self, df: pd.DataFrame,
                             save_path: Optional[str] = None) -> Dict:
        """
        Train direction predictor model.

        Args:
            df: DataFrame with features
            save_path: Path to save trained model

        Returns:
            Training metrics
        """
        logger.info("Training direction predictor...")

        # Prepare training data
        X, y = self.direction_model.prepare_training_data(df)

        if len(X) < 100:
            raise ValueError(f"Not enough training samples: {len(X)}. Need at least 100.")

        # Train model
        metrics = self.direction_model.train(X, y)

        # Save model
        if save_path is None:
            save_path = config.MODELS_DIR / "direction_predictor.joblib"

        self.direction_model.save_model(str(save_path))

        # Log feature importance
        importance = self.direction_model.get_feature_importance()
        logger.info("Top 5 features:")
        logger.info(importance.head())

        return metrics

    def train_all_models(self, instrument: str = "EUR_USD",
                        granularity: str = "H4",
                        days_back: int = 1825) -> Dict:
        """
        Complete training pipeline: fetch data, build features, train both models.

        Args:
            instrument: Currency pair
            granularity: Timeframe
            days_back: Number of days of historical data

        Returns:
            Dictionary with training results
        """
        # Fetch data
        df = self.fetch_training_data(instrument, granularity, days_back)

        # Prepare features
        df = self.prepare_features(df, instrument)

        # Train bounce model
        bounce_metrics = self.train_bounce_model(df)

        # Train direction model
        direction_metrics = self.train_direction_model(df)

        results = {
            'instrument': instrument,
            'granularity': granularity,
            'total_bars': len(df),
            'bounce_model': bounce_metrics,
            'direction_model': direction_metrics
        }

        logger.info("="*50)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Bounce Model - Precision: {bounce_metrics['avg_precision']:.3f}, "
                   f"Recall: {bounce_metrics['avg_recall']:.3f}, "
                   f"ROC-AUC: {bounce_metrics['avg_roc_auc']:.3f}")
        logger.info(f"Direction Model - Accuracy: {direction_metrics['avg_accuracy']:.3f}")
        logger.info("="*50)

        return results


def main():
    """Main training function."""
    pipeline = TrainingPipeline()

    # Train models for each instrument
    for instrument in config.INSTRUMENTS:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training models for {instrument}")
            logger.info(f"{'='*60}\n")

            results = pipeline.train_all_models(
                instrument=instrument,
                granularity="H4",
                days_back=1825  # 5 years
            )

        except Exception as e:
            logger.error(f"Error training {instrument}: {e}")
            continue


if __name__ == "__main__":
    main()

